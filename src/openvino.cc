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

#include <inference_engine.hpp>
#include <stdint.h>
#include <mutex>
#include <vector>
#include "openvino_utils.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

//
// OpenVINO Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace openvino {

namespace {

bool
IsNumber(const std::string& str)
{
  return std::find_if(str.begin(), str.end(), [](unsigned char c) {
           return !std::isdigit(c);
         }) == str.end();
}
}  // namespace

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

  TRITONSERVER_Error* PrintModelConfig();
  TRITONSERVER_Error* ParseParameters();
  TRITONSERVER_Error* ParseParameters(const std::string& device);
  TRITONSERVER_Error* LoadCpuExtensions(
      triton::common::TritonJson::Value& params);
  TRITONSERVER_Error* ParseBoolParameter(
      const std::string& mkey, triton::common::TritonJson::Value& params,
      bool* setting);
  TRITONSERVER_Error* ParseParameter(
      const std::string& mkey, triton::common::TritonJson::Value& params,
      std::map<std::string, std::string>* device_config);
  TRITONSERVER_Error* ParseParameterHelper(
      const std::string& mkey, std::string* ov_key, std::string* value);

  TRITONSERVER_Error* ConfigureInferenceEngine();

  // Reads the Intermediate Representation(IR) model using `artifact_name`
  // as the name for the model file/directory. Return in `model_path` the
  // full path to the model file, return `network` the CNNNetwork.
  TRITONSERVER_Error* ReadNetwork(
      const std::string& artifact_name, std::string* model_path);

  TRITONSERVER_Error* ValidateConfigureNetwork();
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();

  // Loads the configured model on the target device (currently only CPU) is
  // supported.
  TRITONSERVER_Error* LoadNetwork(
      const std::string& device,
      const std::map<std::string, std::string> network_config);

  // Creates an infer request object on the specified device.
  TRITONSERVER_Error* CreateInferRequest(
      const std::string& device, InferenceEngine::InferRequest* infer_request);

  TRITONSERVER_Error* GetInputsInfo(
      InferenceEngine::InputsDataMap* input_tensor_infos);

  // Whether or not the network is read successfully
  bool NetworkNotRead();
  // Whether or not a executable network is loaded on
  // the specified device.
  bool NetworkNotLoaded(const std::string device_);

  InferenceEngine::CNNNetwork* Network() { return &network_; }
  bool SkipDynamicBatchSize() { return skip_dynamic_batchsize_; }
  bool EnableBatchPadding() { return enable_padding_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();

  // Shared resources among the multiple instances.
  InferenceEngine::Core inference_engine_;
  InferenceEngine::CNNNetwork network_;
  std::map<std::string, InferenceEngine::ExecutableNetwork> executable_network_;
  // Maps device to their respective parameters
  std::map<std::string, std::map<std::string, std::string>> config_;
  bool network_read_;
  bool skip_dynamic_batchsize_;
  bool enable_padding_;
  bool reshape_io_layers_;
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
  catch (const InferenceEngine::Exception& e) {
    return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("ModelState::Create InferenceEngine::Exception: ") + e.what()).c_str());
  }
  catch (...) {
    return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "ModelState::Create exception");
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
    : BackendModel(triton_model), network_read_(false),
      skip_dynamic_batchsize_(false), enable_padding_(false),
      reshape_io_layers_(false)
{
}

TRITONSERVER_Error*
ModelState::PrintModelConfig()
{
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ReadNetwork(
    const std::string& artifact_name, std::string* model_path)
{
  RETURN_ERROR_IF_FALSE(
      NetworkNotRead(), TRITONSERVER_ERROR_INTERNAL,
      std::string("attempt to read model at '") + *model_path +
          "' more than once");

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
        std::string("unable to find '") + *model_path + "' for model '" +
            Name() + "'");
  }

  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      network_, inference_engine_.ReadNetwork(*model_path), "reading network");

  network_read_ = true;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    RETURN_IF_ERROR(ParseBoolParameter(
        "SKIP_OV_DYNAMIC_BATCHSIZE", params, &skip_dynamic_batchsize_));
    RETURN_IF_ERROR(
        ParseBoolParameter("ENABLE_BATCH_PADDING", params, &enable_padding_));
    RETURN_IF_ERROR(
        ParseBoolParameter("RESHAPE_IO_LAYERS", params, &reshape_io_layers_));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ParseParameters(const std::string& device)
{
  // Validate and set parameters
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    if (device == "CPU") {
      RETURN_IF_ERROR(LoadCpuExtensions(params));
      config_[device] = {};
      auto& device_config = config_.at(device);
      RETURN_IF_ERROR(
          ParseParameter("CPU_THREADS_NUM", params, &device_config));
      RETURN_IF_ERROR(ParseParameter("ENFORCE_BF16", params, &device_config));
      RETURN_IF_ERROR(
          ParseParameter("CPU_BIND_THREAD", params, &device_config));
      RETURN_IF_ERROR(
          ParseParameter("CPU_THROUGHPUT_STREAMS", params, &device_config));
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadCpuExtensions(triton::common::TritonJson::Value& params)
{
  std::string cpu_ext_path;
  ReadParameter(params, "CPU_EXTENSION_PATH", &(cpu_ext_path));
  if (!cpu_ext_path.empty()) {
    // CPU (MKLDNN) extensions is loaded as a shared library and passed as a
    // pointer to base extension
    const auto extension_ptr =
        std::make_shared<InferenceEngine::Extension>(
            cpu_ext_path);
    RETURN_IF_OPENVINO_ERROR(
        inference_engine_.AddExtension(extension_ptr),
        " loading custom CPU extensions");
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("CPU (MKLDNN) extensions is loaded") + cpu_ext_path)
            .c_str());
  }

  return nullptr;
}


TRITONSERVER_Error*
ModelState::ParseBoolParameter(
    const std::string& mkey, triton::common::TritonJson::Value& params,
    bool* setting)
{
  std::string value;
  ReadParameter(params, mkey, &(value));
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (value.compare("yes") == 0) {
    *setting = true;
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ParseParameter(
    const std::string& mkey, triton::common::TritonJson::Value& params,
    std::map<std::string, std::string>* device_config)
{
  std::string value;
  ReadParameter(params, mkey, &(value));
  if (!value.empty()) {
    std::string ov_key;
    RETURN_IF_ERROR(ParseParameterHelper(mkey, &ov_key, &value));
    (*device_config)[ov_key] = value;
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ParseParameterHelper(
    const std::string& mkey, std::string* ov_key, std::string* value)
{
  std::transform(
      value->begin(), value->end(), value->begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (mkey.compare("CPU_THREADS_NUM") == 0) {
    if (!IsNumber(*value)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be a non-negative number, got " + *value)
              .c_str());
    }
    *ov_key = CONFIG_KEY(CPU_THREADS_NUM);
  } else if (mkey.compare("ENFORCE_BF16") == 0) {
    if (value->compare("yes") == 0) {
      *value = CONFIG_VALUE(YES);
    } else if (value->compare("no") == 0) {
      *value = CONFIG_VALUE(NO);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be either YES or NO, got " + *value)
              .c_str());
    }
    *ov_key = CONFIG_KEY(ENFORCE_BF16);
  } else if (mkey.compare("CPU_BIND_THREAD") == 0) {
    if (value->compare("yes") == 0) {
      *value = CONFIG_VALUE(YES);
    } else if (value->compare("numa") == 0) {
      *value = CONFIG_VALUE(NUMA);
    } else if (value->compare("no") == 0) {
      *value = CONFIG_VALUE(NO);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be either YES/NUMA/NO, got " + *value)
              .c_str());
    }
    *ov_key = CONFIG_KEY(CPU_BIND_THREAD);
  } else if (mkey.compare("CPU_THROUGHPUT_STREAMS") == 0) {
    if (value->compare("auto") == 0) {
      *value = "CPU_THROUGHPUT_AUTO";
    } else if (value->compare("numa") == 0) {
      *value = "CPU_THROUGHPUT_NUMA";
    } else if (!IsNumber(*value)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be a non-negative number or AUTO/NUMA, got " + *value)
              .c_str());
    }
    *ov_key = CONFIG_KEY(CPU_THROUGHPUT_STREAMS);
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("the parameter '") + mkey +
         "' is not yet supported by openvino backend")
            .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ConfigureInferenceEngine()
{
  for (auto&& item : config_) {
    RETURN_IF_OPENVINO_ERROR(
        inference_engine_.SetConfig(item.second, item.first),
        "configuring inference engine");
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadNetwork(
    const std::string& device,
    const std::map<std::string, std::string> network_config)
{
  RETURN_ERROR_IF_FALSE(
      NetworkNotLoaded(device), TRITONSERVER_ERROR_INTERNAL,
      std::string("attempt to load model '") + Name() + "' on device '" +
          device + "' more than once");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("InferenceEngine: ") +
       InferenceEngine::GetInferenceEngineVersion()->description)
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Device info: \n") +
       ConvertVersionMapToString(inference_engine_.GetVersions(device)))
          .c_str());

  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      executable_network_[device],
      inference_engine_.LoadNetwork(network_, device, network_config),
      "loading network");

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateInferRequest(
    const std::string& device, InferenceEngine::InferRequest* infer_request)
{
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      *infer_request, executable_network_[device].CreateInferRequest(),
      "creating infer request object");

  return nullptr;
}

bool
ModelState::NetworkNotRead()
{
  return !network_read_;
}

bool
ModelState::NetworkNotLoaded(const std::string device)
{
  auto itr = executable_network_.find(device);
  return (itr == executable_network_.end());
}

TRITONSERVER_Error*
ModelState::ValidateConfigureNetwork()
{
  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_config_.Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }
  }

  RETURN_IF_ERROR(ValidateInputs(expected_input_cnt));
  RETURN_IF_ERROR(ValidateOutputs());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateInputs(const size_t expected_input_cnt)
{
  InferenceEngine::InputsDataMap input_tensor_infos;
  RETURN_IF_ERROR(GetInputsInfo(&input_tensor_infos));
  if (input_tensor_infos.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " + std::to_string(input_tensor_infos.size()))
            .c_str());
  }

  std::set<std::string> input_tensor_names;
  for (const auto& input_tensor_info : input_tensor_infos) {
    input_tensor_names.insert(input_tensor_info.first);
  }

  InferenceEngine::ICNNNetwork::InputShapes model_shapes;
  if (reshape_io_layers_) {
    RETURN_IF_OPENVINO_ASSIGN_ERROR(
        model_shapes, network_.getInputShapes(),
        "retrieving original shapes from the network");
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &ios));
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
           io_name + "' for model '" + Name() + "'")
              .c_str());
    }
    RETURN_IF_OPENVINO_ERROR(
        iit->second->setPrecision(openvino_precision),
        std::string("setting precision for " + io_name).c_str());

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    if (reshape_io_layers_) {
      int index = (MaxBatchSize() != 0) ? 1 : 0;
      for (const auto dim : dims) {
        model_shapes[io_name][index++] = dim;
      }
    } else {
      RETURN_IF_ERROR(CompareDimsSupported(
          Name(), io_name, iit->second->getTensorDesc().getDims(), dims,
          MaxBatchSize(), false /* compare_exact */));
    }
  }

  if (reshape_io_layers_) {
    network_.reshape(model_shapes);
  }

  // Configuring the network to handle the max_batch_size
  RETURN_IF_OPENVINO_ERROR(
      SetBatchSize(MaxBatchSize(), &network_), "setting max batch size");
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateOutputs()
{
  InferenceEngine::OutputsDataMap output_tensor_infos;
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      output_tensor_infos, network_.getOutputsInfo(),
      "getting output infos for validation");
  std::set<std::string> output_tensor_names;
  for (const auto& output_tensor_info : output_tensor_infos) {
    output_tensor_names.insert(output_tensor_info.first);
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &ios));
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
           io_name + "' for model '" + Name() + "'")
              .c_str());
    }
    RETURN_IF_OPENVINO_ERROR(
        iit->second->setPrecision(openvino_precision),
        std::string("setting precision for " + io_name).c_str());

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
        Name(), io_name, iit->second->getTensorDesc().getDims(), dims,
        MaxBatchSize(), true /* compare_exact */));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::GetInputsInfo(InferenceEngine::InputsDataMap* input_tensor_infos)
{
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      *input_tensor_infos, network_.getInputsInfo(), "getting input infos");

  return nullptr;
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
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* SetBatch(const int batch_size);
  TRITONSERVER_Error* Infer(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      std::vector<const char*>* input_names);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  TRITONSERVER_Error* ValidateOutputBatchSize(
      std::vector<int64_t>* output_shape);

  ModelState* model_state_;

  // The full path to the model file.
  std::string model_path_;

  std::string device_;
  InferenceEngine::InferRequest infer_request_;
  std::map<std::string, InferenceEngine::Blob::Ptr> input_blobs_;

  size_t batch_pad_size_;
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
  catch (const InferenceEngine::Exception& e) {
    return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("ModelState::Create InferenceEngine::Exception: ") + e.what()).c_str());
  }
  catch (...) {
    return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "ModelState::Create exception");
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), device_("CPU"), batch_pad_size_(0)
{
  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    throw triton::backend::BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', openVINO backend supports only CPU device")
            .c_str()));
  }

  if (model_state_->NetworkNotRead()) {
    THROW_IF_BACKEND_INSTANCE_ERROR(
        model_state_->ReadNetwork(ArtifactFilename(), &model_path_));
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ParseParameters());
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ValidateConfigureNetwork());
  }

  if (model_state_->NetworkNotLoaded(device_)) {
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ParseParameters(device_));
    // enable dynamic batching in the network
    std::map<std::string, std::string> network_config;
    if ((model_state_->MaxBatchSize() != 0) &&
        (!model_state_->SkipDynamicBatchSize())) {
      network_config
          [InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] =
              InferenceEngine::PluginConfigParams::YES;
    }
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ConfigureInferenceEngine());
    THROW_IF_BACKEND_INSTANCE_ERROR(
        model_state_->LoadNetwork(device_, network_config));
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->CreateInferRequest(device_, &infer_request_));
}

ModelInstanceState::~ModelInstanceState()
{
  for (auto itr : input_blobs_) {
    itr.second->deallocate();
  }
}

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
                  "null request given to openVINO backend for '" + Name() + "'")
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
      if (total_batch_size != (size_t)max_batch_size) {
        if (model_state_->EnableBatchPadding()) {
          batch_pad_size_ = max_batch_size - total_batch_size;
        } else {
          RequestsRespondWithError(
              requests, request_count,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  std::string(
                      "expected requests with batch size '" +
                      std::to_string(max_batch_size) + "', got '" +
                      std::to_string(total_batch_size) +
                      "'... this error can be avoided by setting "
                      "'ENABLE_BATCH_PADDING' parameter in model configuration "
                      "to 'YES' at a performance cost.")
                      .c_str()));
          return;
        }
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

  if (!model_state_->SkipDynamicBatchSize()) {
    // Sets the new batch size before issuing the inference.
    if (max_batch_size != 0) {
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          &responses, request_count, SetBatch(total_batch_size));
    }
  }

  std::vector<const char*> input_names;
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &input_names);
  // Request to retrieve all model outputs.
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
      }
    }
  }

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  // Run...
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      &responses, request_count, Infer(&responses, request_count));

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
          "failed to send openvino backend response");
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

TRITONSERVER_Error*
ModelInstanceState::SetBatch(const int batch_size)
{
  RETURN_IF_OPENVINO_ERROR(
      infer_request_.SetBatch(batch_size), "setting batch size");

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::Infer(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  RETURN_IF_OPENVINO_ERROR(infer_request_.Infer(), "running inference");

  return nullptr;
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    std::vector<const char*>* input_names)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  InferenceEngine::InputsDataMap input_tensor_infos;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      model_state_->GetInputsInfo(&input_tensor_infos));

  BackendInputCollector collector(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream());
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));

    input_names->emplace_back(input_name);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    const int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    if (batch_pad_size_ != 0) {
      if (input_blobs_.find(input_name) == input_blobs_.end()) {
        input_blobs_[input_name] =
            GetInputBlob(input_tensor_infos[input_name]->getTensorDesc());
        input_blobs_[input_name]->allocate();
      }
      auto data_blob = input_blobs_[input_name];
      if ((size_t)batchn_byte_size != data_blob->byteSize()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("padding input with ") +
             std::to_string(batch_pad_size_) +
             " additional batches to match max_batch_size, send requests with "
             "batch_size equal to max_batch_size for better performance.")
                .c_str());
      }
      auto dest = (data_blob->buffer()).as<char*>();
      memset(dest, 0, data_blob->byteSize());
      collector.ProcessTensor(
          input_name, dest, data_blob->byteSize(), TRITONSERVER_MEMORY_CPU, 0);
      RESPOND_ALL_AND_RETURN_IF_OPENVINO_ERROR(
          responses, request_count,
          infer_request_.SetBlob(input_name, data_blob),
          "setting the input tensor data");
    } else {
      const char* input_buffer;
      size_t buffer_byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
            responses, request_count,
      collector.ProcessTensor(
          input_name, nullptr, 0, {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}},
          &input_buffer, &buffer_byte_size, &memory_type, &memory_type_id));
      if (memory_type == TRITONSERVER_MEMORY_GPU) {
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            responses, request_count,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to get input buffer in CPU memory"));
      }

      if ((uint64_t)batchn_byte_size != buffer_byte_size) {
        SendErrorForResponses(
            responses, request_count,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                std::string(
                    "expected " + std::to_string(batchn_byte_size) +
                    " bytes of data in input buffer, got " +
                    std::to_string(buffer_byte_size) + " bytes.")
                    .c_str()));
      }

      // Set the input blob to the buffer without allocating any new memory
      auto data_blob = WrapInputBufferToBlob(
          input_tensor_infos[input_name]->getTensorDesc(), input_buffer,
          batchn_byte_size);
      RESPOND_ALL_AND_RETURN_IF_OPENVINO_ERROR(
          responses, request_count,
          infer_request_.SetBlob(input_name, data_blob),
          "setting the input tensor data");
    }
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

  bool cuda_copy = false;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];

    InferenceEngine::Blob::Ptr output_blob;
    RESPOND_ALL_AND_RETURN_IF_OPENVINO_ASSIGN_ERROR(
        output_blob, responses, request_count, infer_request_.GetBlob(name),
        "reading output tensor blob ");
    auto output_shape =
        ConvertToSignedShape(output_blob->getTensorDesc().getDims());

    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count, ValidateOutputBatchSize(&output_shape));

    auto const mem_locker = output_blob->cbuffer();
    responder.ProcessTensor(
        name,
        ConvertFromOpenVINOPrecision(
            output_blob->getTensorDesc().getPrecision()),
        output_shape, mem_locker.as<const char*>(), TRITONSERVER_MEMORY_CPU, 0);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputBatchSize(std::vector<int64_t>* output_shape)
{
  auto mbs = model_state_->MaxBatchSize();
  if (mbs == 0) {
    return nullptr;
  } else if (
      ((*output_shape)[0] != mbs) &&
      ((size_t)(*output_shape)[0] != (mbs - batch_pad_size_))) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string(
             "expected the batch size of openvino model output to be ") +
         std::to_string(mbs) + ", got " + std::to_string((*output_shape)[0]))
            .c_str());
  }

  (*output_shape)[0] = mbs - batch_pad_size_;

  return nullptr;
}

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

  RETURN_IF_ERROR(model_state->PrintModelConfig());

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
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::openvino
