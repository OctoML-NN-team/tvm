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

/**
 * \file
 * \brief Simple JSON runtime for Apple BNNS primitives
 */

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace ::tvm::runtime;
using namespace ::tvm::runtime::json;
//using namespace ::tvm::runtime::contrib::MPS;

/**
 * Main entry point to MPS runtime
 */
class MPSJSONRuntime : public JSONRuntimeBase {
 public:
  MPSJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const override { return "mps_json"; }

  void Init(const Array<NDArray>& consts) override {
    std::cout << "MPSJSONRuntime::Init" << std::endl;
    //ICHECK_EQ(consts.size(), const_idx_.size())
    //    << "The number of input constants must match the number of required.";

    //SetupConstants(consts);
    //BindInputsAndOutputs();
    //AllocateIntermediateTensors();
    //BuildEngine();
  }

  void Run() override {
    std::cout << "MPSJSONRuntime::Run()" << std::endl;
    // Wrap external handler into BNNS tensor representation
    //auto bind_ext_hdl_to_tensor = [this](uint32_t eid) {
    //  const auto& ext_dlt = *data_entry_[eid];
    //  auto& bnns_tensor = tensors_eid_[eid];
    //  bnns_tensor->set_data_hdl(ext_dlt.data);
    //};

    //// Bind all input/output external data object into internal abstractions
    //for (const auto& eid : input_var_eid_) bind_ext_hdl_to_tensor(eid);
    //for (const auto& out_entity : outputs_) bind_ext_hdl_to_tensor(EntryID(out_entity));

    //// Invoke primitives in topological order
    //for (const auto& prim : primitives_) prim->execute();
  }

 private:
  /** Make corresponding input/output tensor stubs */
  void BindInputsAndOutputs() {
    std::cout << "MPSJSONRuntime::BindInputsAndOutputs" << std::endl;
    //tensors_eid_.resize(data_entry_.size());
    //auto createTensor = [&](JSONGraphNodeEntry entry) {
    //  auto node = nodes_[entry.id_];
    //  auto dlshape = node.GetOpShape()[entry.index_];
    //  auto dltype = node.GetOpDataType()[entry.index_];
    //  void* data = nullptr;
    //  if (data_entry_[entry.id_] != nullptr) data = data_entry_[entry.id_]->data;
    //  tensors_eid_[entry.id_] = std::make_shared<BNNS::Tensor>(
    //      BNNS::Shape{dlshape.begin(), dlshape.end()}, convertToBNNS(dltype), data);
    //};

    //for (auto& id : input_nodes_) {
    //  auto eid = JSONGraphNodeEntry(id, 0);
    //  createTensor(eid);
    //}

    //for (auto entry : outputs_) {
    //  createTensor(entry);
    //}
  }

  /** Allocate intermediate tensors */
  void AllocateIntermediateTensors() {
    std::cout << "MPSJSONRuntime::AllocateIntermediateTensors" << std::endl;
    //for (int i = 0; i < nodes_.size(); ++i) {
    //  auto eid = JSONGraphNodeEntry(i, 0);
    //  if (tensors_eid_[eid.id_] != nullptr) continue;
    //  auto node = nodes_[i];
    //  auto dlshape = node.GetOpShape()[0];
    //  auto dltype = node.GetOpDataType()[0];
    //  tensors_eid_[eid.id_] = std::make_shared<BNNS::Tensor>(
    //      BNNS::Shape{dlshape.begin(), dlshape.end()}, convertToBNNS(dltype), nullptr);
    //  tensors_eid_[eid.id_]->allocate_memory();
    //}
  }

  // Build up the engine based on the input graph.
  void BuildEngine() {
    std::cout << "MPSJSONRuntime::BuildEngine" << std::endl;
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  void Conv2d(const size_t& nid, const bool has_bias = false,
              const std::string activation_type = "none") {
      std::cout << "MPS::Conv2d" << std::endl;
  }

  /** Collection of all primitives in topological order */
  //std::vector<std::shared_ptr<BNNS::Primitive>> primitives_;

  /** Vector with BNNS tensors. Index of tensor matched with
   *  corresponding EntryID from base JSONRuntimeBase. */
  //std::vector<TensorPtr> tensors_eid_;
};

runtime::Module MPSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<MPSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MPSJSONRuntimeCreate").set_body_typed(MPSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bnns_json")
    .set_body_typed(MPSJSONRuntime::LoadFromBinary<MPSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

