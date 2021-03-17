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

#include "rpc_trace.h"
#include <os/signpost.h>

namespace tvm {
namespace runtime {

os_log_t g_log = os_log_create("tvm.runtime.rpc", "all_RPC");

struct trace_dmn_imp {
  os_log_t log;
};

struct trace_ctx_imp {
  os_log_t log;
  const char* name;
  os_signpost_id_t id;
};

trace_dmn trace_domain_create(const char* domain_name, const char* group_name) {
  trace_dmn res;
  auto res_impl = reinterpret_cast<trace_dmn_imp*>(&res);
  res_impl->log = os_log_create(domain_name, group_name);
  return res;
}

trace_ctx trace_ctx_create(trace_dmn &domain, const char* name) {
  trace_ctx res;
  auto res_impl = reinterpret_cast<trace_ctx_imp*>(&res);
  res_impl->log = reinterpret_cast<trace_dmn_imp*>(&domain)->log;
  res_impl->name = name;
  res_impl->id = OS_SIGNPOST_ID_EXCLUSIVE;
  return res;
}

void trace_region_begin(trace_ctx &ctx) {
  auto ctx_impl = reinterpret_cast<trace_ctx_imp*>(&ctx);
  uint8_t _Alignas(16) OS_LOG_UNINITIALIZED _os_fmt_buf[__builtin_os_log_format_buffer_size("")];
  _os_signpost_emit_with_name_impl(&__dso_handle, ctx_impl->log, OS_SIGNPOST_INTERVAL_BEGIN,
                                   ctx_impl->id, ctx_impl->name, "",
                                   (uint8_t *)__builtin_os_log_format(_os_fmt_buf, ""),
                                   (uint32_t)sizeof(_os_fmt_buf));
}

void trace_region_end(trace_ctx &ctx) {
  auto ctx_impl = reinterpret_cast<trace_ctx_imp*>(&ctx);
  uint8_t _Alignas(16) OS_LOG_UNINITIALIZED _os_fmt_buf[__builtin_os_log_format_buffer_size("")];
  _os_signpost_emit_with_name_impl(&__dso_handle, ctx_impl->log, OS_SIGNPOST_INTERVAL_END,
                                   ctx_impl->id, ctx_impl->name, "",
                                   (uint8_t *)__builtin_os_log_format(_os_fmt_buf, ""),
                                   (uint32_t)sizeof(_os_fmt_buf));
}

void trace_event_emit(trace_ctx ctx) {
  auto ctx_impl = reinterpret_cast<trace_ctx_imp*>(&ctx);
  uint8_t _Alignas(16) OS_LOG_UNINITIALIZED _os_fmt_buf[__builtin_os_log_format_buffer_size("")];
  _os_signpost_emit_with_name_impl(&__dso_handle, ctx_impl->log, OS_SIGNPOST_EVENT,
                                   ctx_impl->id, ctx_impl->name, "",
                                   (uint8_t *)__builtin_os_log_format(_os_fmt_buf, ""),
                                   (uint32_t)sizeof(_os_fmt_buf));
}

}
}
