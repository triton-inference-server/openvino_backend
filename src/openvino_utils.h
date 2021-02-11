// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <inference_engine.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace openvino {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

std::string OpenVINOPrecision(InferenceEngine::Precision openvino_precision);

TRITONSERVER_DataType ConvertFromOpenVINOPrecision(
    InferenceEngine::Precision openvino_precision);

InferenceEngine::Precision ConvertToOpenVINOPrecision(
    TRITONSERVER_DataType data_type);
InferenceEngine::Precision ConvertToOpenVINOPrecision(
    const std::string& data_type_str);

InferenceEngine::Precision ModelConfigDataTypeToOpenVINOPrecision(
    const std::string& data_type_str);
std::string OpenVINOPrecisionToModelConfigDataType(
    InferenceEngine::Precision data_type);

TRITONSERVER_Error* CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<size_t>& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact);

void SetBatchSize(
    const size_t batch_size, InferenceEngine::CNNNetwork* network);
bool AdjustShapesBatch(
    InferenceEngine::ICNNNetwork::InputShapes& shapes, const size_t batch_size,
    const InferenceEngine::InputsDataMap& input_info);

}}}  // namespace triton::backend::openvino
