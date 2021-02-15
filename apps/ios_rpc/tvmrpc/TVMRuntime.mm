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
 * \file TVMRuntime.mm
 */
#include "TVMRuntime.h"
#include <tvm/runtime/module.h>
#include <runtime/rpc/rpc_channel.h>
#include <runtime/rpc/rpc_endpoint.h>
#include <runtime/file_utils.h>

#include <fstream>

namespace tvm {
namespace runtime {

std::shared_ptr<RPCEndpoint> RPCConnect(std::string url, int port, std::string key,
                                        TVMArgs init_seq);

}
}

namespace dmlc {
// Override logging mechanism
void CustomLogMessage::Log(const std::string& msg) { NSLog(@"%s", msg.c_str()); }
}  // namespace dmlc

namespace tvm {
namespace runtime {

class NSStreamChannel final : public RPCChannel {
 public:
  explicit NSStreamChannel(NSOutputStream* stream) : stream_(stream) {}

  size_t Send(const void* data, size_t size) final {
    ssize_t nbytes = [stream_ write:reinterpret_cast<const uint8_t*>(data) maxLength:size];
    if (nbytes < 0) {
      NSLog(@"%@", [stream_ streamError].localizedDescription);
      throw dmlc::Error("Stream error");
    }
    return nbytes;
  }

  size_t Recv(void* data, size_t size) final {
    LOG(FATAL) << "Do not allow explicit receive for";
    return 0;
  }

 private:
  NSOutputStream* stream_;
};

FEventHandler CreateServerEventHandler(NSOutputStream* outputStream, std::string name,
                                       std::string remote_key) {
  std::unique_ptr<NSStreamChannel> ch(new NSStreamChannel(outputStream));
  std::shared_ptr<RPCEndpoint> sess = RPCEndpoint::Create(std::move(ch), name, remote_key);
  return [sess](const std::string& in_bytes, int flag) {
    return sess->ServerAsyncIOEventHandler(in_bytes, flag);
  };
}

// Runtime environment
struct RPCEnv {
 public:
  RPCEnv() {
    NSString* path = NSTemporaryDirectory();
    base_ = [path UTF8String];
    if (base_[base_.length() - 1] != '/') {
      base_ = base_ + '/';
    }
  }
  // Get Path.
  std::string GetPath(const std::string& file_name) { return base_ + file_name; }

 private:
  std::string base_;
};

void LaunchSyncServer() {
  // only load dylib from frameworks.
  NSBundle* bundle = [NSBundle mainBundle];
  NSString* base = [bundle privateFrameworksPath];
  NSString* path = [base stringByAppendingPathComponent:@"tvm/rpc_config.txt"];
  std::string name = [path UTF8String];
  std::ifstream fs(name, std::ios::in);
  std::string url, key;
  int port;
  ICHECK(fs >> url >> port >> key) << "Invalid RPC config file " << name;
  RPCConnect(url, port, "server:" + key, TVMArgs(nullptr, nullptr, 0))->ServerLoop();
}

TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([](TVMArgs args, TVMRetValue* rv) {
  static RPCEnv env;
  *rv = env.GetPath(args[0]);
});

TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  std::string fmt = GetFileFormat(name, "");
  NSString* base;
  base = NSTemporaryDirectory();

  NSString* path =
      [base stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
  name = [path UTF8String];
  *rv = Module::LoadFromFile(name, fmt);
  LOG(INFO) << "Load module from " << name << " ...";
});
}  // namespace runtime
}  // namespace tvm

@implementation TVMRuntime

+ (void)launchSyncServer {
  tvm::runtime::LaunchSyncServer();
}

@end
