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

#include "openvino_utils.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace openvino {

std::string
ConvertVersionMapToString(
    const std::map<std::string, InferenceEngine::Version>& version_map)
{
  std::string result;
  for (const auto& it : version_map) {
    result += (it.first) + ":" + (it.second.description) + "\n ";
  }
  return result;
}

std::string
OpenVINOPrecisionToString(InferenceEngine::Precision openvino_precision)
{
  switch (openvino_precision) {
    case InferenceEngine::Precision::UNSPECIFIED:
      return "UNSPECIFIED";
    case InferenceEngine::Precision::MIXED:
      return "MIXED";
    case InferenceEngine::Precision::FP32:
      return "FLOAT";
    case InferenceEngine::Precision::FP16:
      return "FLOAT16";
    case InferenceEngine::Precision::BF16:
      return "BFLOAT16";
    case InferenceEngine::Precision::Q78:
      return "Q78";
    case InferenceEngine::Precision::I16:
      return "INT16";
    case InferenceEngine::Precision::U8:
      return "UINT8";
    case InferenceEngine::Precision::I8:
      return "INT8";
    case InferenceEngine::Precision::U16:
      return "UINT16";
    case InferenceEngine::Precision::I32:
      return "INT32";
    case InferenceEngine::Precision::U32:
      return "UINT32";
    case InferenceEngine::Precision::I64:
      return "INT64";
    case InferenceEngine::Precision::U64:
      return "UINT64";
    case InferenceEngine::Precision::BIN:
      return "BIN";
    case InferenceEngine::Precision::BOOL:
      return "BOOL";
    case InferenceEngine::Precision::CUSTOM:
      return "CUSTOM";
    default:
      break;
  }

  return "UNDEFINED";
}

TRITONSERVER_DataType ConvertFromOpenVINOPrecision(
    InferenceEngine::Precision openvino_precision);

TRITONSERVER_DataType
ConvertFromOpenVINOPrecision(InferenceEngine::Precision openvino_precision)
{
  switch (openvino_precision) {
    case InferenceEngine::Precision::FP32:
      // maps to c type float (4 bytes)
      return TRITONSERVER_TYPE_FP32;
    case InferenceEngine::Precision::U8:
      return TRITONSERVER_TYPE_UINT8;
    case InferenceEngine::Precision::I8:
      return TRITONSERVER_TYPE_INT8;
    case InferenceEngine::Precision::U16:
      return TRITONSERVER_TYPE_UINT16;
    case InferenceEngine::Precision::I16:
      return TRITONSERVER_TYPE_INT16;
    case InferenceEngine::Precision::I32:
      return TRITONSERVER_TYPE_INT32;
    case InferenceEngine::Precision::I64:
      return TRITONSERVER_TYPE_INT64;
    case InferenceEngine::Precision::BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case InferenceEngine::Precision::FP16:
      return TRITONSERVER_TYPE_FP16;
    case InferenceEngine::Precision::U32:
      return TRITONSERVER_TYPE_UINT32;
    case InferenceEngine::Precision::U64:
      return TRITONSERVER_TYPE_UINT64;
    // The following types are not supported:
    // Unspecified value. Used by default
    case InferenceEngine::Precision::UNSPECIFIED:
    // Mixed value. Can be received from network. No applicable for tensors
    case InferenceEngine::Precision::MIXED:
    // 16bit floating point value, 8 bit for exponent, 7 bit for mantisa
    case InferenceEngine::Precision::BF16:
    // 16bit specific signed fixed point precision
    case InferenceEngine::Precision::Q78:
    // 1bit integer value
    case InferenceEngine::Precision::BIN:
    // custom precision has it's own name and size of elements
    case InferenceEngine::Precision::CUSTOM:
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

InferenceEngine::Precision
ConvertToOpenVINOPrecision(TRITONSERVER_DataType data_type)
{
  switch (data_type) {
    case TRITONSERVER_TYPE_UINT8:
      return InferenceEngine::Precision::U8;
    case TRITONSERVER_TYPE_UINT16:
      return InferenceEngine::Precision::U16;
    case TRITONSERVER_TYPE_UINT32:
      return InferenceEngine::Precision::U32;
    case TRITONSERVER_TYPE_UINT64:
      return InferenceEngine::Precision::U64;
    case TRITONSERVER_TYPE_INT8:
      return InferenceEngine::Precision::I8;
    case TRITONSERVER_TYPE_INT16:
      return InferenceEngine::Precision::I16;
    case TRITONSERVER_TYPE_INT32:
      return InferenceEngine::Precision::I32;
    case TRITONSERVER_TYPE_INT64:
      return InferenceEngine::Precision::I64;
    case TRITONSERVER_TYPE_FP16:
      return InferenceEngine::Precision::FP16;
    case TRITONSERVER_TYPE_FP32:
      return InferenceEngine::Precision::FP32;
    case TRITONSERVER_TYPE_BOOL:
      return InferenceEngine::Precision::BOOL;
    default:
      break;
  }

  return InferenceEngine::Precision::UNSPECIFIED;
}

InferenceEngine::Precision
ConvertToOpenVINOPrecision(const std::string& data_type_str)
{
  TRITONSERVER_DataType data_type =
      TRITONSERVER_StringToDataType(data_type_str.c_str());
  return ConvertToOpenVINOPrecision(data_type);
}

InferenceEngine::Precision
ModelConfigDataTypeToOpenVINOPrecision(const std::string& data_type_str)
{
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return InferenceEngine::Precision::UNSPECIFIED;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "BOOL") {
    return InferenceEngine::Precision::BOOL;
  } else if (dtype == "UINT8") {
    return InferenceEngine::Precision::U8;
  } else if (dtype == "UINT16") {
    return InferenceEngine::Precision::U16;
  } else if (dtype == "UINT32") {
    return InferenceEngine::Precision::U32;
  } else if (dtype == "UINT64") {
    return InferenceEngine::Precision::U64;
  } else if (dtype == "INT8") {
    return InferenceEngine::Precision::I8;
  } else if (dtype == "INT16") {
    return InferenceEngine::Precision::I16;
  } else if (dtype == "INT32") {
    return InferenceEngine::Precision::I32;
  } else if (dtype == "INT64") {
    return InferenceEngine::Precision::I64;
    //} else if (dtype == "FP16") {
    //  return InferenceEngine::Precision::FP16;
  } else if (dtype == "FP32") {
    return InferenceEngine::Precision::FP32;
  }

  return InferenceEngine::Precision::UNSPECIFIED;
}

std::string
OpenVINOPrecisionToModelConfigDataType(InferenceEngine::Precision data_type)
{
  if (data_type == InferenceEngine::Precision::BOOL) {
    return "TYPE_BOOL";
  } else if (data_type == InferenceEngine::Precision::U8) {
    return "TYPE_UINT8";
  } else if (data_type == InferenceEngine::Precision::U16) {
    return "TYPE_UINT16";
  } else if (data_type == InferenceEngine::Precision::U32) {
    return "TYPE_UINT32";
  } else if (data_type == InferenceEngine::Precision::U64) {
    return "TYPE_UINT64";
  } else if (data_type == InferenceEngine::Precision::I8) {
    return "TYPE_INT8";
  } else if (data_type == InferenceEngine::Precision::I16) {
    return "TYPE_INT16";
  } else if (data_type == InferenceEngine::Precision::I32) {
    return "TYPE_INT32";
  } else if (data_type == InferenceEngine::Precision::I64) {
    return "TYPE_INT64";
  } else if (data_type == InferenceEngine::Precision::FP16) {
    return "TYPE_FP16";
  } else if (data_type == InferenceEngine::Precision::FP32) {
    return "TYPE_FP32";
  }

  return "TYPE_INVALID";
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const std::vector<size_t>& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact)
{
  // TODO: OpenVINO backend does not support the dynamic shapes as of now.
  // We can use RESIZE_BILINEAR preProcess in InputInfo to support dynamic
  // shapes in future.
  for (const auto& dim : dims) {
    RETURN_ERROR_IF_TRUE(
        (dim == -1), TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': provides -1 dim (shape " + ShapeToString(dims) +
            "), openvino "
            "currently does not support dynamic shapes.");
  }

  // If the model configuration expects batching support in the model,
  // then the openvino first dimension will be reshaped hence should not
  // be compared.
  const bool supports_batching = (max_batch_size > 0);
  if (supports_batching) {
    RETURN_ERROR_IF_TRUE(
        (model_shape.size() == 0), TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': for the model to support batching the shape should have at "
            "least 1 dimension");

    std::vector<int64_t> full_dims;
    full_dims.reserve(1 + dims.size());
    full_dims.push_back(max_batch_size);
    full_dims.insert(full_dims.end(), dims.begin(), dims.end());

    bool succ = (model_shape.size() == (size_t)full_dims.size());
    if (succ) {
      for (size_t i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (compare_exact || (i != 0)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " +
            ShapeToString(ConvertToSignedShape(model_shape)) +
            ") but the model configuration specifies " +
            std::to_string(full_dims.size()) +
            " dimensions (an initial batch dimension because max_batch_size "
            "> 0 followed by the explicit tensor shape, making complete "
            "shape " +
            ShapeToString(full_dims) + ")");
  } else {
    // ! supports_batching
    bool succ = (model_shape.size() == dims.size());
    if (succ) {
      for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        succ &= (model_dim == dims[i]);
      }
    }

    RETURN_ERROR_IF_TRUE(
        !succ, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("model '") + model_name + "', tensor '" + tensor_name +
            "': the model expects " + std::to_string(model_shape.size()) +
            " dimensions (shape " +
            ShapeToString(ConvertToSignedShape(model_shape)) +
            ") but the model configuration specifies " +
            std::to_string(dims.size()) + " dimensions (shape " +
            ShapeToString(dims) + ")");
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ReadParameter(
    triton::common::TritonJson::Value& params, const std::string& key,
    std::string* param)
{
  triton::common::TritonJson::Value value;
  RETURN_ERROR_IF_FALSE(
      params.Find(key.c_str(), &value), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration is missing the parameter ") + key);
  RETURN_IF_ERROR(value.MemberAsString("string_value", param));
  return nullptr;  // success
}

void
SetBatchSize(const size_t batch_size, InferenceEngine::CNNNetwork* network)
{
  if (batch_size == 0) {
    return;
  }

  InferenceEngine::InputsDataMap inputInfo(network->getInputsInfo());
  InferenceEngine::ICNNNetwork::InputShapes shapes = network->getInputShapes();

  bool reshape = false;
  reshape |= AdjustShapesBatch(shapes, batch_size, inputInfo);

  if (reshape) {
    network->reshape(shapes);
  }

  // It is advisable not to use the setBatchSize API.
  // std::cout << "Reshaping network: " << getShapesString(shapes) << std::endl;
  // network->setBatchSize(batch_size);
}

bool
AdjustShapesBatch(
    InferenceEngine::ICNNNetwork::InputShapes& shapes, const size_t batch_size,
    const InferenceEngine::InputsDataMap& input_info)
{
  bool updated = false;
  for (auto& item : input_info) {
    auto layout = item.second->getTensorDesc().getLayout();

    int batch_index = -1;
    if ((layout == InferenceEngine::Layout::NCHW) ||
        (layout == InferenceEngine::Layout::NCDHW) ||
        (layout == InferenceEngine::Layout::NHWC) ||
        (layout == InferenceEngine::Layout::NDHWC) ||
        (layout == InferenceEngine::Layout::NC)) {
      batch_index = 0;
    } else if (layout == InferenceEngine::Layout::CN) {
      batch_index = 1;
    }
    if ((batch_index != -1) &&
        (shapes.at(item.first).at(batch_index) != batch_size)) {
      shapes[item.first][batch_index] = batch_size;
      updated = true;
    }
  }
  return updated;
}

std::vector<int64_t>
ConvertToSignedShape(const std::vector<size_t> shape)
{
  return std::vector<int64_t>{shape.begin(), shape.end()};
}

InferenceEngine::Blob::Ptr
WrapInputBufferToBlob(
    const InferenceEngine::TensorDesc& tensor_desc,
    const void* input_buffer_ptr, size_t input_buffer_size)
{
  auto precision{tensor_desc.getPrecision()};
  if (precision == InferenceEngine::Precision::BOOL) {
    return InferenceEngine::make_shared_blob<bool>(
        tensor_desc,
        reinterpret_cast<bool*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::U8) {
    return InferenceEngine::make_shared_blob<uint8_t>(
        tensor_desc,
        reinterpret_cast<uint8_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::U16) {
    return InferenceEngine::make_shared_blob<uint16_t>(
        tensor_desc,
        reinterpret_cast<uint16_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::U32) {
    return InferenceEngine::make_shared_blob<uint32_t>(
        tensor_desc,
        reinterpret_cast<uint32_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::U64) {
    return InferenceEngine::make_shared_blob<uint64_t>(
        tensor_desc,
        reinterpret_cast<uint64_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::I8) {
    return InferenceEngine::make_shared_blob<int8_t>(
        tensor_desc,
        reinterpret_cast<int8_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::I16) {
    return InferenceEngine::make_shared_blob<int16_t>(
        tensor_desc,
        reinterpret_cast<int16_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::I32) {
    return InferenceEngine::make_shared_blob<int32_t>(
        tensor_desc,
        reinterpret_cast<int32_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else if (precision == InferenceEngine::Precision::I64) {
    return InferenceEngine::make_shared_blob<int64_t>(
        tensor_desc,
        reinterpret_cast<int64_t*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  } else {
    return InferenceEngine::make_shared_blob<float>(
        tensor_desc,
        reinterpret_cast<float*>(const_cast<void*>(input_buffer_ptr)),
        input_buffer_size);
  }
}


}}}  // namespace triton::backend::openvino
