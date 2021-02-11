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

#include <stdint.h>
#include <mutex>
#include <vector>
#include "openvino_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

//
// OpenVINO Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace openvino {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Reads the Intermediate Representation(IR) model using `artifact_name`
  // as the name for the model file/directory. Return in `model_path` the
  // full path to the model file, return `network` the CNNNetwork.
  TRITONSERVER_Error* ReadNetwork(
      const std::string& artifact_name, std::string* model_path,
      InferenceEngine::CNNNetwork* network);

  // Loads the configured model on the target device (currently only CPU) is
  // supported.
  TRITONSERVER_Error* LoadNetwork(
      InferenceEngine::CNNNetwork& network, std::string& device,
      const std::map<std::string, std::string> network_config,
      InferenceEngine::ExecutableNetwork* executable_network);

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  // All the instances of the model will share Core.
  InferenceEngine::Core inference_core_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
}

TRITONSERVER_Error*
ModelState::ReadNetwork(
    const std::string& artifact_name, std::string* model_path,
    InferenceEngine::CNNNetwork* network)
{
  // Find the IR file that describes the model itself. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.xml").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.xml";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  *network = inference_core_.ReadNetwork(*model_path);

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::LoadNetwork(
    InferenceEngine::CNNNetwork& network, std::string& device,
    const std::map<std::string, std::string> network_config,
    InferenceEngine::ExecutableNetwork* executable_network)
{
  *executable_network =
      inference_core_.LoadNetwork(network, device, network_config);

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // TODO: Add the support for auto completing config for this backend.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for openvino backend")
          .c_str());

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

#if 0
  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);
#endif

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* ValidateConfigureNetwork();
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
#if 0
  void OrtRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count,
      const std::vector<const char*>& input_names,
      const std::vector<const char*>& output_names);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      bool* cuda_copy);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
#endif

  ModelState* model_state_;

  // The full path to the model file.
  std::string model_path_;

  std::string device_;

  InferenceEngine::CNNNetwork network_;
  InferenceEngine::ExecutableNetwork executable_network_;
  InferenceEngine::InferRequest infer_request_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_("CPU")
{
  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    throw triton::backend::BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', openVINO backend supports only CPU device")
            .c_str()));
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->ReadNetwork(ArtifactFilename(), &model_path_, &network_));

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateConfigureNetwork());

  // enable dynamic batching in the network
  std::map<std::string, std::string> network_config;
  if (model_state_->MaxBatchSize() != 0) {
    network_config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] =
        InferenceEngine::PluginConfigParams::YES;
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadNetwork(
      network_, device_, network_config, &executable_network_));

  infer_request_ = executable_network_.CreateInferRequest();
}

TRITONSERVER_Error*
ModelInstanceState::ValidateConfigureNetwork()
{
  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state_->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }
  }

  RETURN_IF_ERROR(ValidateInputs(expected_input_cnt));
  RETURN_IF_ERROR(ValidateOutputs());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  auto input_tensor_infos{network_.getInputsInfo()};
  if (input_tensor_infos.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " + std::to_string(input_tensor_infos.size()))
            .c_str());
  }

  std::set<std::string> input_tensor_names;
  for (const auto& input_tensor_info : input_tensor_infos) {
    input_tensor_names.insert(input_tensor_info.first);
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = input_tensor_infos.find(io_name);
    if (iit == input_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, input_tensor_names));
    }

    auto openvino_precision = ModelConfigDataTypeToOpenVINOPrecision(io_dtype);
    if (openvino_precision == InferenceEngine::Precision::UNSPECIFIED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
    iit->second->setPrecision(openvino_precision);

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }
    RETURN_IF_ERROR(CompareDimsSupported(
        model_state_->Name(), io_name, iit->second->getTensorDesc().getDims(),
        dims, model_state_->MaxBatchSize(), false /* compare_exact */));
  }

  // Configuring the network to handle the max_batch_size
  SetBatchSize(model_state_->MaxBatchSize(), &network_);

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  auto output_tensor_infos{network_.getOutputsInfo()};

  std::set<std::string> output_tensor_names;
  for (const auto& output_tensor_info : output_tensor_infos) {
    output_tensor_names.insert(output_tensor_info.first);
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = output_tensor_infos.find(io_name);
    if (iit == output_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto openvino_precision = ModelConfigDataTypeToOpenVINOPrecision(io_dtype);
    if (openvino_precision == InferenceEngine::Precision::UNSPECIFIED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    }
    iit->second->setPrecision(openvino_precision);

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }
    RETURN_IF_ERROR(CompareDimsSupported(
        model_state_->Name(), io_name, iit->second->getTensorDesc().getDims(),
        dims, model_state_->MaxBatchSize(), true /* compare_exact */));
  }

  return nullptr;  // success
}

#if 0
void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to ONNX Runtime backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Use scoped class to clean up ORT tensors and other resources that
  // need to persist until ORT run completes.
  struct ScopedCleanup {
    ScopedCleanup(ModelInstanceState* ctx) : ctx_(ctx) {}
    ~ScopedCleanup()
    {
      if (ctx_ != nullptr) {
        ctx_->ReleaseOrtRunResources();
      }
    }
    ModelInstanceState* ctx_;
  } io_tensor_wrapper(this);

  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream());
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &cuda_copy);

  // Request to retrieve all model outputs. 'output_names' and
  // 'output_tensors_' are parallel vectors and so must be kept in
  // sync. [TODO] should collect only the outputs needed by some
  // request.
  std::vector<const char*> output_names;
  {
    triton::common::TritonJson::Value ios;
    TRITONSERVER_Error* err =
        model_state_->ModelConfig().MemberAsArray("output", &ios);
    if (err == nullptr) {
      for (size_t i = 0; i < ios.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        err = ios.IndexAsObject(i, &io);
        if (err != nullptr) {
          break;
        }

        // Use names from ModelConfig by reference since the model
        // config will persist longer than this inference execution.
        const char* io_name;
        size_t io_name_len;
        err = io.MemberAsString("name", &io_name, &io_name_len);
        if (err != nullptr) {
          break;
        }

        output_names.emplace_back(io_name);
        output_tensors_.emplace_back(nullptr);
      }
    }

    if (err != nullptr) {
      SendErrorForResponses(&responses, request_count, err);
      output_names.clear();
    }
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  OrtRun(&responses, request_count, input_names, output_names);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  ReadOutputTensors(
      total_batch_size, output_names, requests, request_count, &responses);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send onnxruntime backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

void
ModelInstanceState::OrtRun(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count, const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names)
{
  OrtStatus* status = ort_api->Run(
      session_, NULL /* run options */, input_names.data(),
      (const OrtValue* const*)input_tensors_.data(), input_tensors_.size(),
      output_names.data(), output_names.size(), output_tensors_.data());

  if (status != nullptr) {
    OrtErrorCode code = ort_api->GetErrorCode(status);
    std::string msg = ort_api->GetErrorMessage(status);
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("openvino execute failure ") +
             std::to_string(code) + ": " + msg)
                .c_str()));
  }
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    input_names->emplace_back(input_name);
    input_tensors_.emplace_back(nullptr);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    // [TODO] currently ONNX Runtime only recognize input data on CPU
    // https://github.com/microsoft/onnxruntime/issues/1621
    if (input_datatype != TRITONSERVER_TYPE_BYTES) {
      // The input must be in contiguous CPU memory. Use a pinned
      // memory if possible for the case where the inputs are being
      // provided in GPU memory.
      //
      // [TODO] a couple of optimizations are possible here. 1) if we
      // know that all data for this input across all requests was not
      // in GPU memory, then we could just use regular CPU memory and
      // not pinned memory. 2) if there is a single request and for
      // this input the data is already in contiguous CPU memory then
      // we don't need to copy at all.
      const int64_t batchn_byte_size =
          GetByteSize(input_datatype, batchn_shape);

      BackendMemory* input_memory;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          BackendMemory::Create(
              model_state_->TritonMemoryManager(),
              {BackendMemory::AllocationType::CPU_PINNED_POOL,
               BackendMemory::AllocationType::CPU},
              0 /* memory_type_id */, batchn_byte_size, &input_memory));
      input_tensor_memories_.push_back(input_memory);

      TRITONSERVER_MemoryType input_memtype = input_memory->MemoryType();
      char* input_buffer = input_memory->MemoryPtr();

      // Create ORT Tensor
      const OrtMemoryInfo* allocator_info;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->AllocatorGetInfo(allocator_, &allocator_info));
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->CreateTensorWithDataAsOrtValue(
              allocator_info, (void*)input_buffer, batchn_byte_size,
              batchn_shape.data(), batchn_shape.size(),
              ConvertToOnnxDataType(input_datatype), &input_tensors_.back()));

      collector->ProcessTensor(
          input_name, input_buffer, batchn_byte_size, input_memtype, 0);
    } else {
      // For BYTES input, we need to convert the serialized string
      // representation into what is required for ORT. ORT expects a
      // vector of char*, one for each element. For each tensor we get
      // a copy of the data in a contiguous CPU buffer and then
      // in-place modify that from the Triton
      // <int32_len><bytes><int32_len><bytes>... serialization into a
      // <bytes><null-terminator><bytes><null-terminator>... serialization
      // and then initialize 'string_ptrs' to point to each <bytes>.
      std::vector<const char*> string_ptrs;

      SetStringInputTensor(
          requests, request_count, responses, input_name, &string_ptrs,
          cuda_copy);

      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->CreateTensorAsOrtValue(
              allocator_, batchn_shape.data(), batchn_shape.size(),
              ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->FillStringTensor(
              input_tensors_.back(), string_ptrs.data(), string_ptrs.size()));
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
}

void
ModelInstanceState::SetStringInputTensor(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
    std::vector<const char*>* string_ptrs, bool* cuda_copy)
{
  size_t total_byte_size = 0;
  std::vector<size_t> expected_byte_sizes;
  std::vector<size_t> expected_element_cnts;
  expected_byte_sizes.reserve(request_count);
  expected_element_cnts.reserve(request_count);
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &((*responses)[ridx]),
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in));

    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            in, nullptr, nullptr, &input_shape, &input_dims_count,
            &input_byte_size, nullptr));

    // Skip input in this request if error response has already been sent.
    if ((*responses)[ridx] == nullptr) {
      expected_byte_sizes.push_back(0);
      expected_element_cnts.push_back(0);
    } else {
      expected_element_cnts.push_back(
          GetElementCount(input_shape, input_dims_count));
      expected_byte_sizes.push_back(input_byte_size);
    }

    total_byte_size += expected_byte_sizes.back();
  }

  // For string input, the copy to contiguous buffer is needed because ORT
  // expects elements to be C strings thus we need to modify input buffer.
  // Reserve one more byte at the end of input_buffer to ensure last
  // element of String data can become valid C string.
  BackendMemory* input_memory;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      BackendMemory::Create(
          model_state_->TritonMemoryManager(),
          {BackendMemory::AllocationType::CPU_PINNED_POOL,
           BackendMemory::AllocationType::CPU},
          0 /* memory_type_id */, total_byte_size + 1, &input_memory));
  input_tensor_memories_.push_back(input_memory);

  const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
  char* input_buffer = input_memory->MemoryPtr();

  size_t buffer_offset = 0;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    TRITONSERVER_Error* err =
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in);
    if ((err == nullptr) && ((*responses)[ridx] != nullptr)) {
      uint32_t input_buffer_count;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONBACKEND_InputProperties(
              in, nullptr, nullptr, nullptr, nullptr, nullptr,
              &input_buffer_count));

      size_t input_offset = 0;
      for (size_t idx = 0; idx < input_buffer_count; ++idx) {
        const void* src_buffer;
        size_t src_byte_size;
        TRITONSERVER_MemoryType src_memory_type;
        int64_t src_memory_type_id;
        err = TRITONBACKEND_InputBuffer(
            in, idx, &src_buffer, &src_byte_size, &src_memory_type,
            &src_memory_type_id);
        if (err == nullptr) {
          if ((input_offset + src_byte_size) > expected_byte_sizes[ridx]) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("buffer size for input '") + input_name +
                 "' exceeds batch byte size " +
                 std::to_string(expected_byte_sizes[ridx]))
                    .c_str());
          } else {
            bool cuda_used = false;
            err = CopyBuffer(
                input_name, src_memory_type, src_memory_type_id, mem_type, 0,
                src_byte_size, src_buffer,
                input_buffer + buffer_offset + input_offset, CudaStream(),
                &cuda_used);
            *cuda_copy |= cuda_used;
          }
        }

        if (err == nullptr) {
          input_offset += src_byte_size;
        } else {
          break;
        }
      }
    }

    if (err != nullptr) {
      if ((*responses)[ridx] != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), err);
      }

      TRITONSERVER_ErrorDelete(err);
    }

    buffer_offset += expected_byte_sizes[ridx];
  }

  // Modify input buffer and set string expected by ORT
  SetStringInputBuffer(
      input_name, expected_byte_sizes, expected_element_cnts, responses,
      input_buffer, string_ptrs);
  input_buffer[total_byte_size] = 0;
}

void
ModelInstanceState::SetStringInputBuffer(
    const std::string& input_name,
    const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
    std::vector<const char*>* string_ptrs)
{
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    size_t element_cnt = 0;
    if ((*responses)[idx] != nullptr) {
      size_t remaining_bytes = expected_byte_size;
      char* data_content = input_buffer + buffer_copy_offset;
      // Continue if the remaining bytes may still contain size info
      while (remaining_bytes >= sizeof(uint32_t)) {
        if (element_cnt >= expected_element_cnt) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[idx]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unexpected number of string elements ") +
                   std::to_string(element_cnt + 1) + " for inference input '" +
                   input_name + "', expecting " +
                   std::to_string(expected_element_cnt))
                      .c_str()));
          break;
        }

        const uint32_t len = *(reinterpret_cast<const uint32_t*>(data_content));
        remaining_bytes -= sizeof(uint32_t);
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *data_content = 0;
        data_content = data_content + sizeof(uint32_t);
        if (len > remaining_bytes) {
          RESPOND_AND_SET_NULL_IF_ERROR(
              &((*responses)[idx]),
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("incomplete string data for inference input '") +
                   input_name + "', expecting string of length " +
                   std::to_string(len) + " but only " +
                   std::to_string(remaining_bytes) + " bytes available")
                      .c_str()));
          break;
        } else {
          string_ptrs->push_back(data_content);
          element_cnt++;
          data_content = data_content + len;
          remaining_bytes -= len;
        }
      }
    }

    FillStringData(string_ptrs, expected_element_cnt - element_cnt);
    buffer_copy_offset += expected_byte_size;
  }
}

void
ModelInstanceState::FillStringData(
    std::vector<const char*>* string_ptrs, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_ptrs->push_back(empty);
  }
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;
  std::vector<std::vector<char>> string_buffers;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];

    OrtValue* output_tensor = output_tensors_[idx];
    if (output_tensor == nullptr) {
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("output tensor '") + name + "' is not found")
                  .c_str()));
    }

    // Get output type and shape
    OrtTypeInfo* typeinfo;
    RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
        responses, request_count,
        ort_api->GetTypeInfo(output_tensor, &typeinfo));
    std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

    const OrtTensorTypeAndShapeInfo* type_and_shape;
    RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
        responses, request_count,
        ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

    size_t num_dims;
    RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
        responses, request_count,
        ort_api->GetDimensionsCount(type_and_shape, &num_dims));

    std::vector<int64_t> batchn_shape(num_dims);
    RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
        responses, request_count,
        ort_api->GetDimensions(
            type_and_shape, batchn_shape.data(), batchn_shape.size()));

    ONNXTensorElementDataType type;
    RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
        responses, request_count,
        ort_api->GetTensorElementType(type_and_shape, &type));

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      const size_t element_count = GetElementCount(batchn_shape);
      size_t total_length = 0;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetStringTensorDataLength(output_tensor, &total_length));

      string_buffers.emplace_back(std::vector<char>(total_length));
      auto content = string_buffers.back().data();
      std::vector<size_t> offsets(element_count + 1);
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetStringTensorContent(
              output_tensor, content, total_length, offsets.data(),
              element_count));
      // Mark "passed end byte offset"
      offsets[element_count] = total_length;

      cuda_copy |= SetStringOutputBuffer(
          name, content, offsets.data(), &batchn_shape, requests, request_count,
          responses);
    } else {
      // Fixed size data type...
      char* output_buffer = nullptr;
      RESPOND_ALL_AND_RETURN_IF_ORT_ERROR(
          responses, request_count,
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));

      // [TODO] currently ONNX output data are always on CPU
      // https://github.com/microsoft/onnxruntime/issues/1621
      responder.ProcessTensor(
          name, ConvertFromOnnxDataType(type), batchn_shape, output_buffer,
          TRITONSERVER_MEMORY_CPU, 0);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

}

bool
ModelInstanceState::SetStringOutputBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];

    // batchn_shape holds the shape of the entire tensor batch. When
    // batching is enabled override the first batch dimension with each
    // requests batch size (reusing for efficiency).
    if (model_state_->MaxBatchSize() > 0) {
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
      const int64_t* shape;
      TRITONBACKEND_InputProperties(
          input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
      (*batchn_shape)[0] = shape[0];
    }

    const size_t expected_element_cnt = GetElementCount(*batchn_shape);

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just skip it
    // in the 'content'.
    bool need_output = false;
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* req_output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &req_output_name));
          if ((response != nullptr) && (req_output_name == name)) {
            need_output = true;
            break;
          }
        }
      }
    }

    if (need_output) {
      TRITONBACKEND_Output* response_output;
      TRITONSERVER_Error* err = TRITONBACKEND_ResponseOutput(
          response, &response_output, name.c_str(), TRITONSERVER_TYPE_BYTES,
          batchn_shape->data(), batchn_shape->size());
      if (err == nullptr) {
        // Calculate expected byte size in advance using string offsets
        const size_t data_byte_size =
            offsets[element_idx + expected_element_cnt] - offsets[element_idx];
        const size_t expected_byte_size =
            data_byte_size + sizeof(uint32_t) * expected_element_cnt;

        TRITONSERVER_MemoryType actual_memory_type =
            TRITONSERVER_MEMORY_CPU_PINNED;
        int64_t actual_memory_type_id = 0;
        void* buffer;
        err = TRITONBACKEND_OutputBuffer(
            response_output, &buffer, expected_byte_size, &actual_memory_type,
            &actual_memory_type_id);
        if (err == nullptr) {
          bool cuda_used = false;
          size_t copied_byte_size = 0;
          for (size_t e = 0; e < expected_element_cnt; ++e) {
            const uint32_t len =
                offsets[element_idx + e + 1] - offsets[element_idx + e];
            // Prepend size of the string
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, sizeof(uint32_t),
                static_cast<const void*>(&len),
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += sizeof(uint32_t);

            // Copy raw string content
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, len, content + offsets[element_idx + e],
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += len;
          }
        }
      }

      RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}
#endif

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  //  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::openvino
