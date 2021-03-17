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
 * \file rpc_trace.h
 * \brief Universal trace API on top of os specific performance counter.
 */
#ifndef TVM_RUNTIME_RPC_RPC_TRACE_H_
#define TVM_RUNTIME_RPC_RPC_TRACE_H_

namespace tvm {
namespace runtime {

struct trace_dmn {
  void* reserved[4];
};

struct trace_ctx {
  void* reserved[4];
};

trace_dmn trace_domain_create(const char* domain_name, const char* group_name);
trace_ctx trace_ctx_create(trace_dmn& domain, const char* name);

void trace_region_begin(trace_ctx &ctx);
void trace_region_end(trace_ctx &ctx);
void trace_event_emit(trace_ctx &ctx);


/**
 * RAII wrapper for trace region
 */
class TraceRegion {
 public:
  TraceRegion(trace_dmn &dmn, const char* name)
      : ctx_(trace_ctx_create(dmn, name)) {
    trace_region_begin(ctx_);
  }

  ~TraceRegion() {
    trace_region_end(ctx_);
  }
 private:
  trace_ctx ctx_;
};


#ifndef __FUNCTION_NAME__
# ifdef WIN32   //WINDOWS
#  define __FUNCTION_NAME__   __FUNCTION__
# else          //*NIX
#  define __FUNCTION_NAME__   __func__
# endif
#endif

#define TRACE_SET_DOMAIN();

#define TRACE_REGION(_dmn, _name) \
  TraceRegion __trace_obj_##__COUNT__(_dmn, _name)

#define TRACE_FUNC(_dmn) \
  TRACE_REGION(_dmn, __FUNCTION_NAME__)

}
}
#endif  // TVM_RUNTIME_RPC_RPC_TRACE_H_
