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
 * \brief Simple JSON runtime for Apple MPS primitives
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
#include "../../metal/metal_common.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import <Metal/MTLBlitCommandEncoder.h>
#import <Metal/MTLBuffer.h>
#import <Metal/MTLCommandBuffer.h>
#import <Metal/MTLCommandQueue.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLLibrary.h>

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
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    SetupConstants(consts);
    BindInputsAndOutputs();
    //AllocateIntermediateTensors();
    BuildEngine();
  }

  void Run() override {
    std::cout << "MPSJSONRuntime::Run()" << std::endl;
    // Wrap external handler into MPS tensor representation
    auto bind_ext_hdl_to_tensor = [this](uint32_t eid) {
      id<MTLBuffer> buf = (id<MTLBuffer>)data_entry_[eid]->data;
      auto matrix = matrix_eid_[eid];
      MPSMatrixDescriptor* md =
          [MPSMatrixDescriptor matrixDescriptorWithRows:matrix.rows
                                                columns:matrix.columns
                                               matrices:matrix.matrices
                                               rowBytes:matrix.rowBytes
                                            matrixBytes:matrix.matrixBytes
                                               dataType:matrix.dataType];
      [matrix initWithBuffer:buf descriptor:md];
    };

    // Bind all input/output external data object into internal abstractions
    for (const auto& eid : input_var_eid_) {
        bind_ext_hdl_to_tensor(eid);
    }
    for (const auto& out_entity : outputs_) {
        bind_ext_hdl_to_tensor(EntryID(out_entity));
    }

    for (auto getCb : cbf) {
      auto cb = getCb();
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

 private:
  /** Make corresponding input/output tensor stubs */
  void BindInputsAndOutputs() {
      image_eid_.resize(data_entry_.size());
      matrix_eid_.resize(data_entry_.size());
    std::cout << "MPSJSONRuntime::BindInputsAndOutputs" << std::endl;
    auto createTensor = [&](JSONGraphNodeEntry entry) {
      auto node = nodes_[entry.id_];
      auto dlshape = node.GetOpShape()[entry.index_];
      //auto dltype = node.GetOpDataType()[entry.index_];
      void* data = nullptr;
      if (data_entry_[entry.id_] != nullptr) data = data_entry_[entry.id_]->data;

      MPSDataType dtype = MPSDataTypeFloat32;
      MPSMatrixDescriptor* desc =
          [MPSMatrixDescriptor matrixDescriptorWithRows:dlshape[1]
                                                columns:dlshape[2]
                                               matrices:dlshape[0]
                                               rowBytes:dlshape[2] * sizeof(dtype)
                                            matrixBytes:dlshape[2] * dlshape[1] * sizeof(dtype)
                                               dataType:dtype];
      //[MPSMatrixDescriptor matrixDescriptorWithDimensions:dlshape[1]
      //                                            columns:dlshape[2]
      //                                           rowBytes:dlshape[2] * sizeof(dtype)
      //                                           dataType:dtype];
      auto mw = metal::MetalWorkspace::Global();
      // TODO: It will be necessary to understand id of device!!!
      Device dev = {kDLMetal, 0};
      id<MTLDevice> device = mw->GetDevice(dev);
      if (data != nullptr) {
        id<MTLBuffer>buf = [device newBufferWithBytes:data length:dlshape[0] * dlshape[1] * dlshape[2] * sizeof(dtype) options:MTLResourceStorageModeShared];
        matrix_eid_[entry.id_] = [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
      } else {
        matrix_eid_[entry.id_] = [[MPSMatrix alloc] initWithDevice:device descriptor:desc];
      }
    };

    for (auto& id : input_nodes_) {
      auto entry = JSONGraphNodeEntry(id, 0);
      createTensor(entry);
    }

    for (auto entry : outputs_) {
      createTensor(entry);
    }
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
        auto op_name = node.GetOpName();
        /*if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else*/ if ("nn.batch_matmul" == op_name) {
          MatMul(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  void MatMul(const size_t& nid) {
   auto node = nodes_[nid];

   // Setup attributes.
   auto a_entry = node.GetInputs()[0];
   auto b_entry = node.GetInputs()[1];
   auto dst_entry = JSONGraphNodeEntry(nid, 0);

   int aIdx = a_entry.id_;
   int bIdx = b_entry.id_;
   int dstIdx = dst_entry.id_;
   auto matmul = [aIdx, bIdx, dstIdx, this]() {
    auto mw = metal::MetalWorkspace::Global();
    id<MTLDevice> device = mw->GetDevice(data_entry_[0]->device);
    id<MTLCommandQueue> cq = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [cq commandBuffer];

    MPSMatrixMultiplication* mul_obj = [[MPSMatrixMultiplication alloc] init];
    MPSMatrixMultiplication* sgemm = [mul_obj initWithDevice:device
                                               transposeLeft:0
                                              transposeRight:1
                                                  resultRows:matrix_eid_[aIdx].rows
                                               resultColumns:matrix_eid_[dstIdx].columns
                                             interiorColumns:matrix_eid_[aIdx].columns
                                                       alpha:1.0f
                                                        beta:0.0f];

    [sgemm encodeToCommandBuffer:cb leftMatrix:matrix_eid_[aIdx] rightMatrix:matrix_eid_[bIdx] resultMatrix:matrix_eid_[dstIdx]];
    return cb;
   };
   cbf.push_back(matmul);
  }

  /** Collection of all primitives in topological order */
  std::vector<id<MTLCommandBuffer>> commandBuffers_;
  using CommandBufferFunctor = std::function<id<MTLCommandBuffer>()>;
  std::vector<CommandBufferFunctor> cbf;

  /** Vector with BNNS tensors. Index of tensor matched with
   *  corresponding EntryID from base JSONRuntimeBase. */
  //std::vector<TensorPtr> tensors_eid_;
  std::vector<MPSImage*> image_eid_;
  std::vector<MPSMatrix*> matrix_eid_;
};

runtime::Module MPSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<MPSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MPSJSONRuntimeCreate").set_body_typed(MPSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_mps_json")
    .set_body_typed(MPSJSONRuntime::LoadFromBinary<MPSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

