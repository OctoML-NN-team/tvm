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
// Runtime API
//#include "../../../src/runtime/c_runtime_api.cc"
//#include "../../../src/runtime/cpu_device_api.cc"
//#include "../../../src/runtime/dso_library.cc"
//#include "../../../src/runtime/file_utils.cc"
//#include "../../../src/runtime/library_module.cc"
//#include "../../../src/runtime/metadata_module.cc"
//#include "../../../src/runtime/module.cc"
//#include "../../../src/runtime/ndarray.cc"
//#include "../../../src/runtime/object.cc"
//#include "../../../src/runtime/registry.cc"
//#include "../../../src/runtime/system_library.cc"
//#include "../../../src/runtime/thread_pool.cc"
//#include "../../../src/runtime/threading_backend.cc"
//#include "../../../src/runtime/workspace_pool.cc"
//#include "../../../src/runtime/profiling.cc"
//#include "../../../src/runtime/logging.cc"
//
//// RPC server
//#include "../../../src/runtime/rpc/rpc_channel.cc"
//#include "../../../src/runtime/rpc/rpc_endpoint.cc"
//#include "../../../src/runtime/rpc/rpc_local_session.cc"
//#include "../../../src/runtime/rpc/rpc_module.cc"
//#include "../../../src/runtime/rpc/rpc_server_env.cc"
//#include "../../../src/runtime/rpc/rpc_session.cc"
//#include "../../../src/runtime/rpc/rpc_socket_impl.cc"
//// Graph executor
//#include "../../../src/runtime/graph_executor/graph_executor.cc"
//// Metal
//#include "../../../src/runtime/metal/metal_device_api.mm"
//#include "../../../src/runtime/metal/metal_module.mm"
//// CoreML
//#include "../../../src/runtime/contrib/coreml/coreml_runtime.mm"

namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  throw tvm::runtime::InternalError(file, lineno, message);
}

void LogMessageImpl(const std::string& file, int lineno, const std::string& message) {
  NSLog(@"%s:%d: %s", file.c_str(), lineno, message.c_str());
}
}
}
}  // namespace dmlc

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  std::string base = [NSTemporaryDirectory() UTF8String];
  *rv = base + "/" + name;
});

TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  NSString* base = NSTemporaryDirectory();
  NSString* path =
      [base stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
  name = [path UTF8String];
  *rv = Module::LoadFromFile(name);
  LOG(INFO) << "Load module from " << name << " ...";
});
}  // namespace runtime
}  // namespace tvm

@implementation TVMRuntime

+ (void)launchSyncServer {
  throw "UNIMPLEMENTED";
//  tvm::runtime::LaunchSyncServer();
}

@end
