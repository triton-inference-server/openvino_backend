// Copyright 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
ConvertVersionMapToString(const std::map<std::string, ov::Version>& version_map)
{
  std::string result;
  for (const auto& it : version_map) {
    result += (it.first) + ":" + (it.second.description) + "\n ";
  }
  return result;
}

TRITONSERVER_DataType
ConvertFromOpenVINOElement(ov::element::Type openvino_element)
{
  switch (openvino_element) {
    case ov::element::f32:
      // maps to c type float (4 bytes)
      return TRITONSERVER_TYPE_FP32;
    case ov::element::u8:
      return TRITONSERVER_TYPE_UINT8;
    case ov::element::i8:
      return TRITONSERVER_TYPE_INT8;
    case ov::element::u16:
      return TRITONSERVER_TYPE_UINT16;
    case ov::element::i16:
      return TRITONSERVER_TYPE_INT16;
    case ov::element::i32:
      return TRITONSERVER_TYPE_INT32;
    case ov::element::i64:
      return TRITONSERVER_TYPE_INT64;
    case ov::element::boolean:
      return TRITONSERVER_TYPE_BOOL;
    case ov::element::f16:
      return TRITONSERVER_TYPE_FP16;
    case ov::element::u32:
      return TRITONSERVER_TYPE_UINT32;
    case ov::element::u64:
      return TRITONSERVER_TYPE_UINT64;
    // The following types are not supported:
    // Unspecified value. Used by default
    case ov::element::undefined:
    // Dynamic value
    case ov::element::dynamic:
    // 16bit floating point value, 8 bit for exponent, 7 bit for mantisa
    case ov::element::bf16:
    // 64bit floating point value
    case ov::element::f64:
    // 4bit integer value
    case ov::element::i4:
    // 1bit unsigned integer value
    case ov::element::u1:
    // 4bit unsigned integer value
    case ov::element::u4:
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

ov::element::Type
ConvertToOpenVINOElement(TRITONSERVER_DataType data_type)
{
  switch (data_type) {
    case TRITONSERVER_TYPE_UINT8:
      return ov::element::u8;
    case TRITONSERVER_TYPE_UINT16:
      return ov::element::u16;
    case TRITONSERVER_TYPE_UINT32:
      return ov::element::u32;
    case TRITONSERVER_TYPE_UINT64:
      return ov::element::u64;
    case TRITONSERVER_TYPE_INT8:
      return ov::element::i8;
    case TRITONSERVER_TYPE_INT16:
      return ov::element::i16;
    case TRITONSERVER_TYPE_INT32:
      return ov::element::i32;
    case TRITONSERVER_TYPE_INT64:
      return ov::element::i64;
    case TRITONSERVER_TYPE_FP16:
      return ov::element::f16;
    case TRITONSERVER_TYPE_FP32:
      return ov::element::f32;
    case TRITONSERVER_TYPE_BOOL:
      return ov::element::boolean;
    default:
      break;
  }

  return ov::element::undefined;
}

ov::element::Type
ConvertToOpenVINOElement(const std::string& data_type_str)
{
  TRITONSERVER_DataType data_type =
      TRITONSERVER_StringToDataType(data_type_str.c_str());
  return ConvertToOpenVINOElement(data_type);
}

ov::element::Type
ModelConfigDataTypeToOpenVINOElement(const std::string& data_type_str)
{
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) == 0) {
    const std::string dtype = data_type_str.substr(strlen("TYPE_"));

    if (dtype == "BOOL") {
      return ov::element::boolean;
    } else if (dtype == "UINT8") {
      return ov::element::u8;
    } else if (dtype == "UINT16") {
      return ov::element::u16;
    } else if (dtype == "UINT32") {
      return ov::element::u32;
    } else if (dtype == "UINT64") {
      return ov::element::u64;
    } else if (dtype == "INT8") {
      return ov::element::i8;
    } else if (dtype == "INT16") {
      return ov::element::i16;
    } else if (dtype == "INT32") {
      return ov::element::i32;
    } else if (dtype == "INT64") {
      return ov::element::i64;
    } else if (dtype == "FP16") {
      return ov::element::f16;
    } else if (dtype == "FP32") {
      return ov::element::f32;
    }
  }

  return ov::element::undefined;
}

std::string
OpenVINOElementToModelConfigDataType(const ov::element::Type& data_type)
{
  if (data_type == ov::element::boolean) {
    return "TYPE_BOOL";
  } else if (data_type == ov::element::u8) {
    return "TYPE_UINT8";
  } else if (data_type == ov::element::u16) {
    return "TYPE_UINT16";
  } else if (data_type == ov::element::u32) {
    return "TYPE_UINT32";
  } else if (data_type == ov::element::u64) {
    return "TYPE_UINT64";
  } else if (data_type == ov::element::i8) {
    return "TYPE_INT8";
  } else if (data_type == ov::element::i16) {
    return "TYPE_INT16";
  } else if (data_type == ov::element::i32) {
    return "TYPE_INT32";
  } else if (data_type == ov::element::i64) {
    return "TYPE_INT64";
  } else if (data_type == ov::element::f16) {
    return "TYPE_FP16";
  } else if (data_type == ov::element::f32) {
    return "TYPE_FP32";
  }

  return "TYPE_INVALID";
}

static bool
doesMatch(const ov::Dimension& ov_dim, int64_t config_dim)
{
  if (ov_dim.is_static()) {
    return ov_dim.get_length() == config_dim;
  }
  if (!ov_dim.get_interval().has_upper_bound()) {
    return true;
  }
  return (config_dim < ov_dim.get_max_length()) &&
         (config_dim > ov_dim.get_min_length());
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const ov::PartialShape& model_shape, const std::vector<int64_t>& dims,
    const int max_batch_size, const bool compare_exact)
{
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
        if (compare_exact || ((i != 0) && (full_dims[i] != -1))) {
          succ &= doesMatch(model_shape[i], full_dims[i]);
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
        succ &= doesMatch(model_shape[i], dims[i]);
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

std::vector<int64_t>
ConvertToSignedShape(const ov::PartialShape& shape)
{
  std::vector<int64_t> out;
  for (const auto& dim : shape) {
    out.emplace_back(dim.is_static() ? dim.get_length() : -1);
  }
  return out;
}

}}}  // namespace triton::backend::openvino
