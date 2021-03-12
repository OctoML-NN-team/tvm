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
#ifndef TVM_APPS_CPP_RPC_SERVER_H_
#define TVM_APPS_CPP_RPC_SERVER_H_

#include <string>
#include <future>

#include "tvm/runtime/c_runtime_api.h"
#include "runtime/rpc/rpc_endpoint.h"
#include "runtime/rpc/rpc_socket_impl.h"
#include "support/socket.h"
#include "rpc_server.h"
#include "rpc_tracker_client.h"

namespace tvm {
namespace runtime {

class RPCServer {
 public:
  /*!
   * \brief Constructor.
   */
  RPCServer(std::string host, int port, int port_end, std::string tracker_addr, std::string key, std::string custom_addr)
      : host_(std::move(host)),
        port_(port),
        my_port_(0),
        port_end_(port_end),
        tracker_addr_(std::move(tracker_addr)),
        key_(std::move(key)),
        custom_addr_(std::move(custom_addr)) {}

  /*!
   * \brief Destructor.
   */
  ~RPCServer() {
    try {
      // Free the resources
      tracker_sock_.Close();
      listen_sock_.Close();
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
    
    in_process = true;
    process_loop = {std::async(std::launch::async, &RPCServer::ListenLoopProc, this)};
  }

  void Stop() {
    LOG(INFO) << "Close connection";
    in_process = false;
    process_loop.get();
    listen_sock_.Close();
  }

 private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    TrackerClient tracker(tracker_addr_, key_, custom_addr_);
    while (in_process) {
      support::TCPSocket conn;
      support::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        tracker.TryConnect();
        // step 2: wait for in-coming connections
        AcceptConnection(&tracker, &conn, &addr, &opts);
      } catch (const char* msg) {
        LOG(WARNING) << "Socket exception: " << msg;
        // close tracker resource
        tracker.Close();
        continue;
      } catch (const std::exception& e) {
        // close tracker resource
        tracker.Close();
        LOG(WARNING) << "Exception standard: " << e.what();
        continue;
      }

      // step 3: serving
      ServerLoopProc(conn, addr);
      // close from our side.
      LOG(INFO) << "Socket Connection Closed";
      conn.Close();
    }
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

    while (in_process) {
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
  static void ServerLoopProc(support::TCPSocket sock, support::SockAddr addr) {
    // Server loop
//    const auto env = RPCEnv(); // TODO:
    RPCServerLoop(int(sock.sockfd));
    LOG(INFO) << "Finish serving " << addr.AsString();
//    env.CleanUp();
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
  bool in_process;
  std::future<void> process_loop;
  support::TCPSocket listen_sock_;
  support::TCPSocket tracker_sock_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_SERVER_H_
