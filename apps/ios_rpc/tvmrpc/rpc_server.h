/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file rpc_server.h
 * \brief RPC Server implementation.
 */
#ifndef TVM_APPS_IOS_RPC_SERVER_H_
#define TVM_APPS_IOS_RPC_SERVER_H_

#include <string>
#include <future>
#include <chrono>
#include <dirent.h>

#include "tvm/runtime/c_runtime_api.h"
#include "runtime/rpc/rpc_endpoint.h"
#include "runtime/rpc/rpc_socket_impl.h"
#include "support/socket.h"
#include "rpc_tracker_client.h"

namespace tvm {
namespace runtime {

std::vector<std::string> ListDir(const std::string& dirname) {
  std::vector<std::string> vec;
  DIR* dp = opendir(dirname.c_str());
  if (dp == nullptr) {
    int errsv = errno;
    LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
  }
  dirent* d;
  while ((d = readdir(dp)) != nullptr) {
    std::string filename = d->d_name;
    if (filename != "." && filename != "..") {
      std::string f = dirname;
      if (f[f.length() - 1] != '/') {
        f += '/';
      }
      f += d->d_name;
      vec.push_back(f);
    }
  }
  closedir(dp);
  return vec;
}

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname The name of the directory
 */
void CleanDir(const std::string& dirname) {
  auto files = ListDir(dirname);
  for (const auto& filename : files) {
    std::string file_path = dirname + "/";
    file_path += filename;
    const int ret = std::remove(filename.c_str());
    if (ret != 0) {
      LOG(WARNING) << "Remove file " << filename << " failed";
    }
  }
}

// Runtime environment
struct RPCEnv {
 public:
  RPCEnv(const std::string &base):base_(base) {}
  // Get Path.
  std::string GetPath(const std::string& file_name) { return base_ + file_name; }

  void CleanUp() const {
    CleanDir(base_);
  }
 private:
  std::string base_;
};


/*!
 * \brief RPCServer RPC Server class.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9099
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 */
class RPCServer {
 public:
  /*!
   * \brief Constructor.
   */
  RPCServer(std::string host, int port, int port_end, std::string tracker_addr, std::string key,
            std::string custom_addr, std::string work_dir)
      : host_(std::move(host)),
        port_(port),
        my_port_(0),
        port_end_(port_end),
        tracker_addr_(std::move(tracker_addr)),
        key_(std::move(key)),
        custom_addr_(std::move(custom_addr)),
        work_dir_(std::move(work_dir)),
        tracker_(tracker_addr_, key_, custom_addr_) {}

  /*!
   * \brief Destructor.
   */
  ~RPCServer() {
    try {
      // Free the resources
      listen_sock_.Close();
      tracker_.Close();
    } catch (...) {
    }
  }

  /*!
   * \brief Start Creates the RPC listen process and execution.
   */
  void Start() {
    listen_sock_.Create();
    my_port_ = listen_sock_.TryBindHost(host_, port_, port_end_);
    LOG(INFO) << "bind to " << host_ << ":" << my_port_;
    listen_sock_.Listen(1);
    continue_processing = true;
    proc_ = std::future<void>(std::async(std::launch::async, &RPCServer::ListenLoopProc, this));
  }
  
  void Stop() {
    continue_processing = false;
    tracker_.Close();
  }
    
  void setCompletionCallbacks(std::function<void()> callback_start, std::function<void()> callback_stop) {
    completion_callback_start_ = callback_start;
    completion_callback_stop_ = callback_stop;
  }

 private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    
    while (continue_processing) {
      support::TCPSocket conn;
      support::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        tracker_.TryConnect();
        if (completion_callback_start_)
          completion_callback_start_();
        // step 2: wait for in-coming connections
        AcceptConnection(&tracker_, &conn, &addr, &opts);
      } catch (const char* msg) {
        LOG(WARNING) << "Socket exception: " << msg;
        // close tracker resource
        tracker_.Close();
        continue;
      } catch (const std::exception& e) {
        // close tracker resource
        tracker_.Close();
        LOG(WARNING) << "Exception standard: " << e.what();
        continue;
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      ServerLoopProc(conn, addr, work_dir_);
      auto dur = std::chrono::high_resolution_clock::now() - start_time;

      LOG(INFO) << "Serve Time " << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "ms";

      // close from our side.
      LOG(INFO) << "Socket Connection Closed";
      conn.Close();
    }
    if (completion_callback_stop_)
      completion_callback_stop_();

  }

  /*!
   * \brief AcceptConnection Accepts the RPC Server connection.
   * \param tracker Tracker details.
   * \param conn_sock New connection information.
   * \param addr New connection address information.
   * \param opts Parsed options for socket
   * \param ping_period Timeout for select call waiting
   */
  void AcceptConnection(TrackerClient* tracker, support::TCPSocket* conn_sock,
                        support::SockAddr* addr, std::string* opts, int ping_period = 2) {
    std::set<std::string> old_keyset;
    std::string matchkey;

    // Report resource to tracker and get key
    tracker->ReportResourceAndGetKey(my_port_, &matchkey);

    while (continue_processing) {
      tracker->WaitConnectionAndUpdateKey(listen_sock_, my_port_, ping_period, &matchkey);
      support::TCPSocket conn = listen_sock_.Accept(addr);

      int code = kRPCMagic;
      ICHECK_EQ(conn.RecvAll(&code, sizeof(code)), sizeof(code));
      if (code != kRPCMagic) {
        conn.Close();
        LOG(FATAL) << "Client connected is not TVM RPC server";
        continue;
      }

      int keylen = 0;
      ICHECK_EQ(conn.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));

      const char* CLIENT_HEADER = "client:";
      const char* SERVER_HEADER = "server:";
      std::string expect_header = CLIENT_HEADER + matchkey;
      std::string server_key = SERVER_HEADER + key_;
      if (size_t(keylen) < expect_header.length()) {
        conn.Close();
        LOG(INFO) << "Wrong client header length";
        continue;
      }

      ICHECK_NE(keylen, 0);
      std::string remote_key;
      remote_key.resize(keylen);
      ICHECK_EQ(conn.RecvAll(&remote_key[0], keylen), keylen);

      std::stringstream ssin(remote_key);
      std::string arg0;
      ssin >> arg0;

      if (arg0 != expect_header) {
        code = kRPCMismatch;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        conn.Close();
        LOG(WARNING) << "Mismatch key from" << addr->AsString();
        continue;
      } else {
        code = kRPCSuccess;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        keylen = int(server_key.length());
        ICHECK_EQ(conn.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
        ICHECK_EQ(conn.SendAll(server_key.c_str(), keylen), keylen);
        LOG(INFO) << "Connection success " << addr->AsString();
        ssin >> *opts;
        *conn_sock = conn;
        return;
      }
    }
  }

  /*!
   * \brief ServerLoopProc The Server loop process.
   * \param sock The socket information
   * \param addr The socket address information
   */
  static void ServerLoopProc(support::TCPSocket sock, support::SockAddr addr,
                             std::string work_dir) {
    // Server loop
    const auto env = RPCEnv(work_dir);
    RPCServerLoop(int(sock.sockfd));
    LOG(INFO) << "Finish serving " << addr.AsString();
    env.CleanUp();
  }

  /*!
   * \brief GetTimeOutFromOpts Parse and get the timeout option.
   * \param opts The option string
   */
  int GetTimeOutFromOpts(const std::string& opts) const {
    const std::string option = "-timeout=";

    size_t pos = opts.rfind(option);
    if (pos != std::string::npos) {
      const std::string cmd = opts.substr(pos + option.size());
      ICHECK(support::IsNumber(cmd)) << "Timeout is not valid";
      return std::stoi(cmd);
    }
    return 0;
  }

  std::string host_;
  int port_;
  int my_port_;
  int port_end_;
  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  std::string work_dir_;
  support::TCPSocket listen_sock_;
  TrackerClient tracker_;
  
  bool continue_processing;
  std::future<void> proc_;
  std::function<void()> completion_callback_start_;
  std::function<void()> completion_callback_stop_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_IOS_RPC_SERVER_H_
