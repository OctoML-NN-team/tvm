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

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
      const auto& ext_dlt = *data_entry_[eid];
      auto image = image_eid_[eid];
      [image writeBytes:ext_dlt.data dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth imageIndex:0];
      //bnns_tensor->set_data_hdl(ext_dlt.data);
    };

    // Bind all input/output external data object into internal abstractions
    for (const auto& eid : input_var_eid_) {
        std::cout << "input eid: " << eid << std::endl;
        bind_ext_hdl_to_tensor(eid);
    }
    for (const auto& out_entity : outputs_) bind_ext_hdl_to_tensor(EntryID(out_entity));

    constexpr int src_size = 1 * 3 * 10 * 14;
    const auto& ext_dlt = *data_entry_[0];
    float* f_ptr = static_cast<float*>(ext_dlt.data);
    std::cout << "input: ";
    for (int i = 0; i < src_size; ++i) {
        std::cout << f_ptr[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    float src_check[src_size];
    [image_eid_[0] readBytes:src_check dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth imageIndex:0];
    std::cout << "src_check: ";
    for (int i = 0; i < src_size; ++i) {
        std::cout << src_check[i] << ", ";
    }
    std::cout << std::endl;

    //// Invoke primitives in topological order
    //for (const auto& prim : primitives_) prim->execute();
    for (id<MTLCommandBuffer> cb : commandBuffers_) {
        [cb commit];
        [cb waitUntilCompleted];
        //id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        //[encoder synchronizeResource:image_eid_[EntryID(outputs_[0])].texture];
        //[encoder endEncoding];
        auto eid = EntryID(outputs_[0]);
        const auto& ext_dlt = *data_entry_[eid];
        [image_eid_[eid] readBytes:ext_dlt.data dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth imageIndex:0];
    }
  }

 private:
  /** Make corresponding input/output tensor stubs */
  void BindInputsAndOutputs() {
      image_eid_.resize(data_entry_.size());
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

    std::cout << "BindInputsAndOutputs, inputs: ";
    for (auto& id : input_nodes_) {
      auto entry = JSONGraphNodeEntry(id, 0);
      auto node = nodes_[entry.id_];
      auto dlshape = node.GetOpShape()[entry.index_];
    //  auto dltype = node.GetOpDataType()[entry.index_];
      std::cout << entry.id_ << " (";
      for (auto& s : dlshape) {
          std::cout << s << ", ";
      }
      std::cout << "), ";
      //createTensor(eid);
    }
    std::cout << std::endl;

    std::cout << "BindInputsAndOutputs, outputs: ";
    for (auto entry : outputs_) {
      std::cout << entry.id_ << " (";
      auto node = nodes_[entry.id_];
      auto dlshape = node.GetOpShape()[entry.index_];
      for (auto& s : dlshape) {
          std::cout << s << ", ";
      }
      std::cout << "), ";
      //createTensor(entry);
    }
    std::cout << std::endl;
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
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto wgh_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    auto src_node = nodes_[src_entry.id_];
    auto wgh_node = nodes_[wgh_entry.id_];
    auto dst_node = nodes_[dst_entry.id_];

    auto src_dlshape = node.GetOpShape()[src_entry.index_];
    auto wgh_dlshape = node.GetOpShape()[wgh_entry.index_];
    auto dst_dlshape = node.GetOpShape()[dst_entry.index_];
    auto src_dltype = node.GetOpDataType()[src_entry.index_];
    auto wgh_dltype = node.GetOpDataType()[wgh_entry.index_];
    auto dst_dltype = node.GetOpDataType()[dst_entry.index_];
    void* src_data = nullptr;
    void* wgh_data = nullptr;
    void* dst_data = nullptr;
    if (data_entry_[src_entry.id_] != nullptr) src_data = data_entry_[src_entry.id_]->data;
    if (data_entry_[wgh_entry.id_] != nullptr) wgh_data = data_entry_[wgh_entry.id_]->data;
    if (data_entry_[dst_entry.id_] != nullptr) dst_data = data_entry_[dst_entry.id_]->data;

    //NSArray<id<MTLDevice> >* devs = MTLCopyAllDevices();
    //id<MTLDevice> device = [devs objectAtIndex:1];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> cq = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [cq commandBuffer];
    commandBuffers_.push_back(cb);

    // Create images
  MPSImageDescriptor* s_desc =
      [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                     width:src_dlshape[3]
                                                    height:src_dlshape[2]
                                           featureChannels:src_dlshape[1]];
  MPSImageDescriptor* w_desc =
      [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                     width:wgh_dlshape[3]
                                                    height:wgh_dlshape[2]
                                           featureChannels:wgh_dlshape[1]];
  MPSImageDescriptor* d_desc =
      [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                     width:dst_dlshape[3]
                                                    height:dst_dlshape[2]
                                           featureChannels:dst_dlshape[1]];

  MTLTextureDescriptor *textureDescriptor = [[MTLTextureDescriptor alloc] init];
    // Indicate that each pixel has a blue, green, red, and alpha channel, where each channel is
    // an 8-bit unsigned normalized value (i.e. 0 maps to 0.0 and 255 maps to 1.0)
    textureDescriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
    // Set the pixel dimensions of the texture
    textureDescriptor.width = src_dlshape[3];
    textureDescriptor.height = src_dlshape[2];
  id<MTLTexture> texture = [device newTextureWithDescriptor:textureDescriptor];

    //MPSImage* src = [[MPSImage alloc] initWithDevice:device imageDescriptor:s_desc];
    MPSImage* src = [[MPSImage alloc] initWithTexture:texture featureChannels:src_dlshape[1]];
    //MPSImage* src = [[MPSImage alloc] initWithDevice:device imageDescriptor:s_desc];
    MPSImage* weight = [[MPSImage alloc] initWithDevice:device imageDescriptor:w_desc];
    MPSImage* dst = [[MPSImage alloc] initWithDevice:device imageDescriptor:d_desc];
    image_eid_[src_entry.id_] = src;
    image_eid_[wgh_entry.id_] = weight;
    image_eid_[dst_entry.id_] = dst;
    //[src writeBytes:src_data
    //      dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
    //      imageIndex:0];

    //const auto& src_t = GetBNNSTensor(src_entry);
    //const auto& wgh_t = GetBNNSTensor(wgh_entry);
    //const auto& dst_t = GetBNNSTensor(dst_entry);
    MPSCNNConvolutionDescriptor* conv_desc =
        [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:3
                                                                kernelHeight:3
                                                        inputFeatureChannels:1
                                                       outputFeatureChannels:1];
    [conv_desc setStrideInPixelsX:1];
    [conv_desc setStrideInPixelsY:1];

    MPSCNNConvolution* conv = [[MPSCNNConvolution alloc] initWithDevice:device
                                                  convolutionDescriptor:conv_desc
                                                          kernelWeights:(float*)wgh_data
                                                              biasTerms:nil
                                                                  flags:MPSCNNConvolutionFlagsNone];
    /*if (pad == 0) {
      conv.padding = [MPSNNDefaultPadding paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                                            MPSNNPaddingMethodAlignCentered |
                                                            MPSNNPaddingMethodSizeSame];
    } else if (pad == 1) {
      conv.padding = [MPSNNDefaultPadding paddingWithMethod:MPSNNPaddingMethodAddRemainderToTopLeft |
                                                            MPSNNPaddingMethodAlignCentered |
                                                            MPSNNPaddingMethodSizeValidOnly];
    }*/
    [conv encodeToCommandBuffer:cb sourceImage:src destinationImage:dst];

  }

  /** Collection of all primitives in topological order */
  std::vector<id<MTLCommandBuffer>> commandBuffers_;

  /** Vector with BNNS tensors. Index of tensor matched with
   *  corresponding EntryID from base JSONRuntimeBase. */
  //std::vector<TensorPtr> tensors_eid_;
  std::vector<MPSImage*> image_eid_;
};

runtime::Module MPSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  std::cout << " >>> MPSJSONRuntimeCreate" << std::endl;
  auto n = make_object<MPSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MPSJSONRuntimeCreate").set_body_typed(MPSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_mps_json")
    .set_body_typed(MPSJSONRuntime::LoadFromBinary<MPSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

