// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <openvino/openvino.hpp>
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

// BackendConfiguration
struct BackendConfiguration {
  BackendConfiguration() : default_max_batch_size_(0) {}
  int default_max_batch_size_;
};

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
      std::vector<std::pair<std::string, ov::Any>>* device_config);
  TRITONSERVER_Error* ParseParameterHelper(
      const std::string& mkey, std::string* value,
      std::pair<std::string, ov::Any>* ov_property);

  TRITONSERVER_Error* ConfigureOpenvinoCore();

  // Reads the Intermediate Representation(IR) model using `artifact_name`
  // as the name for the model file/directory. Return in `model_path` the
  // full path to the model file, return `network` the CNNNetwork.
  TRITONSERVER_Error* ReadModel(
      const std::string& artifact_name, std::string* model_path);

  TRITONSERVER_Error* ValidateConfigureModel();
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();

  // Loads the configured model on the target device (currently only CPU) is
  // supported.
  TRITONSERVER_Error* LoadModel(
      const std::string& device,
      const std::pair<std::string, ov::Any>& property);

  // Creates an infer request object on the specified device.
  TRITONSERVER_Error* CreateInferRequest(
      const std::string& device, ov::InferRequest* infer_request);

  // Whether or not the model is read successfully
  bool ModelNotRead();
  // Whether or not a executable model is loaded on
  // the specified device.
  bool ModelNotLoaded(const std::string device);

  bool SkipDynamicBatchSize() { return skip_dynamic_batchsize_; }
  bool EnableBatchPadding() { return enable_padding_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteBatching(
      const std::vector<ov::Output<ov::Node>>& ov_inputs,
      const std::vector<ov::Output<ov::Node>>& ov_outputs);
  TRITONSERVER_Error* AutoCompleteInputOrOutput(
      const char* io_json_obj_name,
      const std::vector<ov::Output<ov::Node>>& ov_ios);

  // Shared resources among the multiple instances.
  ov::Core ov_core_;
  std::shared_ptr<ov::Model> ov_model_;
  std::map<std::string, ov::CompiledModel> compiled_model_;
  // Maps device to their respective parameters
  std::map<std::string, std::vector<std::pair<std::string, ov::Any>>> config_;
  bool model_read_;
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
  catch (const ov::Exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("ModelState::Create ov::Exception: ") + e.what()).c_str());
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
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), model_read_(false),
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
ModelState::ReadModel(const std::string& artifact_name, std::string* model_path)
{
  RETURN_ERROR_IF_FALSE(
      ModelNotRead(), TRITONSERVER_ERROR_INTERNAL,
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
      ov_model_, ov_core_.read_model(*model_path), "reading model");

  model_read_ = true;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  triton::common::TritonJson::Value params;
  bool status = model_config_.Find("parameters", &params);
  if (status) {
    RETURN_IF_ERROR(LoadCpuExtensions(params));
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
      config_[device] = {};
      auto& device_config = config_.at(device);
      RETURN_IF_ERROR(
          ParseParameter("INFERENCE_NUM_THREADS", params, &device_config));
      RETURN_IF_ERROR(
          ParseParameter("COMPILATION_NUM_THREADS", params, &device_config));
      RETURN_IF_ERROR(ParseParameter("HINT_BF16", params, &device_config));
      RETURN_IF_ERROR(ParseParameter("NUM_STREAMS", params, &device_config));
      RETURN_IF_ERROR(
          ParseParameter("PERFORMANCE_HINT", params, &device_config));
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadCpuExtensions(triton::common::TritonJson::Value& params)
{
  std::string cpu_ext_path;
  LOG_IF_ERROR(
      ReadParameter(params, "CPU_EXTENSION_PATH", &(cpu_ext_path)),
      "error when reading parameters");
  if (!cpu_ext_path.empty()) {
    // CPU (MKLDNN) extensions is loaded as a shared library and passed as a
    // pointer to base extension
    RETURN_IF_OPENVINO_ERROR(
        ov_core_.add_extension(cpu_ext_path), " loading custom CPU extensions");
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
  LOG_IF_ERROR(
      ReadParameter(params, mkey, &(value)), "error when reading parameters");
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
    std::vector<std::pair<std::string, ov::Any>>* device_config)
{
  std::string value;
  LOG_IF_ERROR(
      ReadParameter(params, mkey, &(value)), "error when reading parameters");
  if (!value.empty()) {
    std::pair<std::string, ov::Any> ov_property;
    RETURN_IF_ERROR(ParseParameterHelper(mkey, &value, &ov_property));
    device_config->push_back(ov_property);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ParseParameterHelper(
    const std::string& mkey, std::string* value,
    std::pair<std::string, ov::Any>* ov_property)
{
  std::transform(
      value->begin(), value->end(), value->begin(),
      [](unsigned char c) { return std::tolower(c); });
  if (mkey.compare("INFERENCE_NUM_THREADS") == 0) {
    if (!IsNumber(*value)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be a non-negative number, got " + *value)
              .c_str());
    }
    *ov_property = ov::inference_num_threads(std::stoi(*value));
  } else if (mkey.compare("COMPILATION_NUM_THREADS") == 0) {
    if (!IsNumber(*value)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be a non-negative number, got " + *value)
              .c_str());
    }
    *ov_property = ov::compilation_num_threads(std::stoi(*value));
  } else if (mkey.compare("HINT_BF16") == 0) {
    if (value->compare("yes") == 0) {
      *ov_property = ov::hint::inference_precision(ov::element::bf16);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be YES, got " + *value)
              .c_str());
    }
  } else if (mkey.compare("NUM_STREAMS") == 0) {
    if (value->compare("auto") == 0) {
      *ov_property = ov::streams::num(ov::streams::AUTO);
    } else if (value->compare("numa") == 0) {
      *ov_property = ov::streams::num(ov::streams::NUMA);
    } else if (IsNumber(*value)) {
      *ov_property = ov::streams::num(std::stoi(*value));
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be either AUTO/NUMA/<int_value>, got " + *value)
              .c_str());
    }
  } else if (mkey.compare("PERFORMANCE_HINT") == 0) {
    if (value->compare("latency") == 0) {
      *ov_property =
          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY);
    } else if (value->compare("throughput") == 0) {
      *ov_property =
          ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT);
    } else if (value->compare("cumulative_throughput") == 0) {
      *ov_property = ov::hint::performance_mode(
          ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("expected the parameter '") + mkey +
           "' to be LATENCY/THROUGHPUT/CUMULATIVE_THROUGHPUT, got " + *value)
              .c_str());
    }
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
ModelState::ConfigureOpenvinoCore()
{
  for (auto&& item : config_) {
    std::string device_name = item.first;
    std::vector<std::pair<std::string, ov::Any>> properties = item.second;
    for (auto& property : properties) {
      RETURN_IF_OPENVINO_ERROR(
          ov_core_.set_property(device_name, property),
          "configuring openvino core");
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& device, const std::pair<std::string, ov::Any>& property)
{
  RETURN_ERROR_IF_FALSE(
      ModelNotLoaded(device), TRITONSERVER_ERROR_INTERNAL,
      std::string("attempt to load model '") + Name() + "' on device '" +
          device + "' more than once");

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE, (std::string("Openvino runtime: ") +
                                 std::to_string(OPENVINO_VERSION_MAJOR) + "." +
                                 std::to_string(OPENVINO_VERSION_MINOR) + "." +
                                 std::to_string(OPENVINO_VERSION_PATCH))
                                    .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Device info: \n") +
       ConvertVersionMapToString(ov_core_.get_versions(device)))
          .c_str());

  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      compiled_model_[device],
      ov_core_.compile_model(ov_model_, device, property), "loading model");

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateInferRequest(
    const std::string& device, ov::InferRequest* infer_request)
{
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      *infer_request, compiled_model_[device].create_infer_request(),
      "creating infer request object");

  return nullptr;
}

bool
ModelState::ModelNotRead()
{
  return !model_read_;
}

bool
ModelState::ModelNotLoaded(const std::string device)
{
  auto itr = compiled_model_.find(device);
  return (itr == compiled_model_.end());
}

TRITONSERVER_Error*
ModelState::ValidateConfigureModel()
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
  std::vector<ov::Output<ov::Node>> model_inputs;
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      model_inputs, ov_model_->inputs(), "getting input infos");

  if (model_inputs.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " + std::to_string(model_inputs.size()))
            .c_str());
  }

  std::set<std::string> model_inputs_names;
  std::map<std::string, size_t> model_inputs_name_to_index;
  for (size_t i = 0; i < model_inputs.size(); i++) {
    model_inputs_names.insert(model_inputs[i].get_any_name());
    model_inputs_name_to_index[model_inputs[i].get_any_name()] = i;
  }

  ov::preprocess::PrePostProcessor ppp(ov_model_);

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    if (model_inputs_names.find(io_name) == model_inputs_names.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, model_inputs_names));
    }

    auto openvino_element = ModelConfigDataTypeToOpenVINOElement(io_dtype);
    if (openvino_element == ov::element::undefined) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + Name() + "'")
              .c_str());
    }
    RETURN_IF_OPENVINO_ERROR(
        ppp.input(io_name).tensor().set_element_type(openvino_element),
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

    ov::Shape input_shape;
    ov::PartialShape partial_input_shape;
    RETURN_IF_OPENVINO_ASSIGN_ERROR(
        partial_input_shape,
        model_inputs[model_inputs_name_to_index[io_name]].get_partial_shape(),
        ("retrieving original shapes from input " + io_name).c_str());
    if (reshape_io_layers_) {
      int index = (MaxBatchSize() != 0) ? 1 : 0;
      for (const auto dim : dims) {
        if (dim > 0) {
          partial_input_shape[index++] = ov::Dimension(dim);
        } else if (dim == -1) {
          partial_input_shape[index++] = ov::Dimension::dynamic();
        } else {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "openvino backend does not support dimensions values"
              " other than `-1` or positive integers");
        }
      }
      RETURN_IF_OPENVINO_ERROR(
          ppp.input(io_name).tensor().set_shape(partial_input_shape),
          std::string("setting shape for " + io_name).c_str());
    } else {
      RETURN_IF_ERROR(CompareDimsSupported(
          Name(), io_name, partial_input_shape, dims, MaxBatchSize(),
          false /* compare_exact */));
    }

    if (MaxBatchSize()) {
      RETURN_IF_OPENVINO_ERROR(
          ppp.input(io_name).tensor().set_layout("N..."),
          std::string("setting layout for " + io_name).c_str());
    }
  }

  // Model preprocessing
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      ov_model_, ppp.build(), "apply model input preprocessing");

  // Configuring the model to handle the max_batch_size
  if (MaxBatchSize()) {
    RETURN_IF_OPENVINO_ERROR(
        ov::set_batch(ov_model_, MaxBatchSize()), "setting max batch size");
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateOutputs()
{
  std::vector<ov::Output<ov::Node>> model_outputs;
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      model_outputs, ov_model_->outputs(), "getting output infos");

  std::set<std::string> model_outputs_names;
  std::map<std::string, size_t> model_outputs_name_to_index;
  for (size_t i = 0; i < model_outputs.size(); i++) {
    model_outputs_names.insert(model_outputs[i].get_any_name());
    model_outputs_name_to_index[model_outputs[i].get_any_name()] = i;
  }

  ov::preprocess::PrePostProcessor ppp(ov_model_);

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    if (model_outputs_names.find(io_name) == model_outputs_names.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, model_outputs_names));
    }

    auto openvino_element = ModelConfigDataTypeToOpenVINOElement(io_dtype);
    if (openvino_element == ov::element::undefined) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + Name() + "'")
              .c_str());
    }
    RETURN_IF_OPENVINO_ERROR(
        ppp.output(io_name).tensor().set_element_type(openvino_element),
        std::string("setting precision for " + io_name).c_str());

    // If a reshape is provided for the output then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }
    ov::PartialShape output_shape;
    RETURN_IF_OPENVINO_ASSIGN_ERROR(
        output_shape,
        model_outputs[model_outputs_name_to_index[io_name]].get_partial_shape(),
        ("retrieving original shapes from output " + io_name).c_str());
    RETURN_IF_ERROR(CompareDimsSupported(
        Name(), io_name, output_shape, dims, MaxBatchSize(),
        true /* compare_exact */));
  }

  // Model preprocessing
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      ov_model_, ppp.build(), "apply model output preprocessing");

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Read OV model for autocomplete
  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));
  std::string model_path;
  THROW_IF_BACKEND_INSTANCE_ERROR(ReadModel(artifact_name, &model_path));
  model_read_ = false;  // Re-read model after autocomplete

  // Get OV model inputs and outputs
  std::vector<ov::Output<ov::Node>> model_inputs;
  std::vector<ov::Output<ov::Node>> model_outputs;
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      model_inputs, ov_model_->inputs(), "getting input infos");
  RETURN_IF_OPENVINO_ASSIGN_ERROR(
      model_outputs, ov_model_->outputs(), "getting output infos");

  // Autocomplete batching
  RETURN_IF_ERROR(AutoCompleteBatching(model_inputs, model_outputs));
  // Autocomplete input
  RETURN_IF_ERROR(AutoCompleteInputOrOutput("input", model_inputs));
  // Autocomplete output
  RETURN_IF_ERROR(AutoCompleteInputOrOutput("output", model_outputs));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteBatching(
    const std::vector<ov::Output<ov::Node>>& ov_inputs,
    const std::vector<ov::Output<ov::Node>>& ov_outputs)
{
  // Determine batching support from model layout
  bool support_batching = true;
  {
    for (const std::vector<ov::Output<ov::Node>>& ov_ios :
         {ov_inputs, ov_outputs}) {
      for (const ov::Output<ov::Node>& ov_io : ov_ios) {
        // Get layout of the OV input/output
        ov::Layout ov_layout = ov::layout::get_layout(ov_io);
        // Check if this input/output support batching
        if (!ov::layout::has_batch(ov_layout)) {
          support_batching = false;
          break;
        }
      }
      if (!support_batching)
        break;
    }
  }

  if (support_batching) {
    // The model layout support batching
    // Autocomplete max_batch_size
    if (MaxBatchSize() == 0) {
      // Get default_max_batch_size from backend state
      int max_batch_size = 0;
      {
        TRITONBACKEND_Backend* backend;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_ModelBackend(TritonModel(), &backend));
        void* state;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_BackendState(backend, &state));
        max_batch_size = reinterpret_cast<BackendConfiguration*>(state)
                             ->default_max_batch_size_;
      }
      max_batch_size = std::max(max_batch_size, 1);  // max_batch_size >= 1
      // Set max_batch_size
      triton::common::TritonJson::Value max_batch_size_json;
      ModelConfig().Find("max_batch_size", &max_batch_size_json);
      max_batch_size_json.SetInt(max_batch_size);
      SetMaxBatchSize(max_batch_size);
      // Advise user to specify max_batch_size
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string(
               "autofilled max_batch_size to " +
               std::to_string(max_batch_size) + " for model '") +
           Name() +
           "' since batching is supported but no max_batch_size is "
           "specified "
           "in model configuration. Must specify max_batch_size to utilize "
           "autofill with a larger max batch size")
              .c_str());
    }
    // Autocomplete dynamic batching
    if (MaxBatchSize() > 1) {
      triton::common::TritonJson::Value tmp_json;
      bool dynamic_batching_exist =
          ModelConfig().Find("dynamic_batching", &tmp_json);
      bool sequence_batching_exist =
          ModelConfig().Find("sequence_batching", &tmp_json);
      // Add dynamic batching if dynamic and sequence batching not provided
      if (!dynamic_batching_exist && !sequence_batching_exist) {
        triton::common::TritonJson::Value dynamic_batching_json(
            ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERROR(ModelConfig().Add(
            "dynamic_batching", std::move(dynamic_batching_json)));
      }
    }
  } else if (MaxBatchSize() != 0) {
    // The model layout does not support batching but max_batch_size != 0
    // Not all openvino models include proper layout when batching is supported
    // Warn the user about this discrepancy
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("model layout for model ") + Name() +
         " does not support batching while non-zero max_batch_size is "
         "specified")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteInputOrOutput(
    const char* io_json_obj_name,
    const std::vector<ov::Output<ov::Node>>& ov_ios)
{
  // Read current input/output json
  size_t curr_num_ios = 0;
  triton::common::TritonJson::Value curr_ios_json;
  bool ios_exist = ModelConfig().Find(io_json_obj_name, &curr_ios_json);
  if (ios_exist) {
    curr_num_ios += curr_ios_json.ArraySize();
  }

  // Autocomplete inputs/outputs if none is provided
  if (curr_num_ios == 0) {
    // New input/output json to be build
    triton::common::TritonJson::Value new_ios_json(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    // Populate new input/output json from OV inputs/outputs
    for (const ov::Output<ov::Node>& ov_io : ov_ios) {
      // New individual input/output
      triton::common::TritonJson::Value io_json(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      // Populate name
      std::string io_name = ov_io.get_any_name();
      RETURN_IF_ERROR(io_json.AddString("name", io_name));
      // Populate data type
      RETURN_IF_ERROR(io_json.AddString(
          "data_type",
          OpenVINOElementToModelConfigDataType(ov_io.get_element_type())));
      // Find shape
      ov::PartialShape io_shape;
      RETURN_IF_OPENVINO_ASSIGN_ERROR(
          io_shape, ov_io.get_partial_shape(),
          ("retrieving original shapes from" + std::string(io_json_obj_name) +
           " " + io_name)
              .c_str());
      // Populate dims
      triton::common::TritonJson::Value dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_shape.size(); i++) {
        RETURN_IF_ERROR(dims.AppendInt(
            io_shape.is_static() ? io_shape[i].get_length() : -1));
      }
      RETURN_IF_ERROR(io_json.Add("dims", std::move(dims)));
      // Add individual input/output to new input/output
      RETURN_IF_ERROR(new_ios_json.Append(std::move(io_json)));
    }
    // Add new input/output to config
    if (ios_exist) {
      curr_ios_json.Swap(new_ios_json);
    } else {
      ModelConfig().Add(io_json_obj_name, std::move(new_ios_json));
    }
  } else {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping ") + io_json_obj_name +
         " model configuration auto-complete for '" + Name() +
         "': " + io_json_obj_name + " already specified")
            .c_str());
  }
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

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  TRITONSERVER_Error* Infer(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      std::vector<const char*>* input_names);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  TRITONSERVER_Error* ValidateOutputBatchSize(
      std::vector<int64_t>* output_shape);

  ModelState* model_state_;

  std::string device_;
  ov::InferRequest infer_request_;

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
  catch (const ov::Exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("ModelInstanceState::Create ov::Exception: ") + e.what())
            .c_str());
  }
  catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "ModelInstanceState::Create exception");
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
         "', Triton openVINO backend supports only CPU device")
            .c_str()));
  }

  if (model_state_->ModelNotRead()) {
    std::string model_path;
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ParseParameters());
    THROW_IF_BACKEND_INSTANCE_ERROR(
        model_state_->ReadModel(ArtifactFilename(), &model_path));
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ValidateConfigureModel());
  }

  if (model_state_->ModelNotLoaded(device_)) {
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ParseParameters(device_));
    // enable dynamic batching in the model
    std::pair<std::string, ov::Any> property =
        ov::hint::allow_auto_batching(false);
    if ((model_state_->MaxBatchSize() != 0) &&
        (!model_state_->SkipDynamicBatchSize())) {
      property = ov::hint::allow_auto_batching(true);
    }
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->ConfigureOpenvinoCore());
    THROW_IF_BACKEND_INSTANCE_ERROR(model_state_->LoadModel(device_, property));
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(
      model_state_->CreateInferRequest(device_, &infer_request_));
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
  bool all_response_failed = false;

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

  for (size_t i = 0; i < request_count; i++) {
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
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed, err);
      }
      if (!all_response_failed) {
        if (total_batch_size != (size_t)max_batch_size) {
          if (model_state_->EnableBatchPadding()) {
            batch_pad_size_ = max_batch_size - total_batch_size;
          } else {
            RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
                responses, request_count, all_response_failed,
                TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL,
                    std::string(
                        "expected requests with batch size '" +
                        std::to_string(max_batch_size) + "', got '" +
                        std::to_string(total_batch_size) +
                        "'... this error can be avoided by setting "
                        "'ENABLE_BATCH_PADDING' parameter in model "
                        "configuration "
                        "to 'YES' at a performance cost.")
                        .c_str()));
          }
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
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "batch size " + std::to_string(total_batch_size) + " for '" +
                  Name() + "', max allowed is " +
                  std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  std::vector<const char*> input_names;
  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(
            total_batch_size, requests, request_count, &responses,
            &input_names));
  }

  // Request to retrieve all model outputs.
  std::vector<const char*> output_names;
  if (!all_response_failed) {
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
  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        Infer(&responses, request_count));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(
            total_batch_size, output_names, requests, request_count,
            &responses));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // was successful or not.
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

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::Infer(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  RETURN_IF_OPENVINO_ERROR(infer_request_.start_async(), "running inference");
  infer_request_.wait();

  return nullptr;
}

TRITONSERVER_Error*
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
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  BackendInputCollector collector(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    const int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    if (batch_pad_size_ != 0) {
      ov::Tensor input_tensor =
          infer_request_.get_tensor(std::string(input_name));
      if ((size_t)batchn_byte_size != input_tensor.get_byte_size()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("padding input with ") +
             std::to_string(batch_pad_size_) +
             " additional batches to match max_batch_size, send requests with "
             "batch_size equal to max_batch_size for better performance.")
                .c_str());
      }
      char* dest = (char*)input_tensor.data(ov::element::undefined);
      memset(dest, 0, input_tensor.get_byte_size());
      collector.ProcessTensor(
          input_name, dest, input_tensor.get_byte_size(),
          TRITONSERVER_MEMORY_CPU, 0);
    } else {
      char* input_buffer;
      size_t buffer_byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERROR(collector.ProcessTensor(
          input_name, nullptr, 0,
          {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}},
          (const char**)&input_buffer, &buffer_byte_size, &memory_type,
          &memory_type_id));
      if (memory_type == TRITONSERVER_MEMORY_GPU) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "failed to get input buffer in CPU memory"));
      }

      if ((uint64_t)batchn_byte_size != buffer_byte_size) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            std::string(
                "expected " + std::to_string(batchn_byte_size) +
                " bytes of data in input buffer, got " +
                std::to_string(buffer_byte_size) + " bytes.")
                .c_str()));
      }

      // Set the input tensor to the buffer without allocating any new memory
      ov::Tensor input_tensor(
          ConvertToOpenVINOElement(input_datatype),
          std::vector<size_t>(batchn_shape.begin(), batchn_shape.end()),
          input_buffer);
      RETURN_IF_OPENVINO_ERROR(
          infer_request_.set_tensor(std::string(input_name), input_tensor),
          "setting tensor data");
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  bool cuda_copy = false;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];

    ov::Tensor output_tensor = infer_request_.get_tensor(name);
    std::vector<int64_t> output_shape =
        ConvertToSignedShape(output_tensor.get_shape());

    RETURN_IF_ERROR(ValidateOutputBatchSize(&output_shape));

    responder.ProcessTensor(
        name, ConvertFromOpenVINOElement(output_tensor.get_element_type()),
        output_shape, (const char*)output_tensor.data(ov::element::undefined),
        TRITONSERVER_MEMORY_CPU, 0);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

  return nullptr;
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

  // Read backend config message
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));
  // Serialize message
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  // Convert message to JSON
  triton::common::TritonJson::Value backend_config_json;
  TRITONSERVER_Error* err = nullptr;
  if (byte_size != 0) {
    err = backend_config_json.Parse(buffer, byte_size);
  }
  RETURN_IF_ERROR(err);
  // Create backend configuration
  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  // Read command-line arguments from config
  triton::common::TritonJson::Value cmd_json;
  if (backend_config_json.Find("cmdline", &cmd_json)) {
    // Find default-max-batch-size
    triton::common::TritonJson::Value val_json;
    if (cmd_json.Find("default-max-batch-size", &val_json)) {
      // Get default-max-batch-size
      std::string val_str;
      RETURN_IF_ERROR(val_json.AsString(&val_str));
      int val_int;
      RETURN_IF_ERROR(ParseIntValue(val_str, &val_int));
      // Write default-max-batch-size to backend configuration
      lconfig->default_max_batch_size_ = val_int;
    }
  }
  // Set backend configuration
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));
  lconfig.release();

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto config = reinterpret_cast<BackendConfiguration*>(vstate);
  delete config;
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

TRITONSERVER_Error*
TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_GetBackendAttribute: setting attributes");
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeAddPreferredInstanceGroup(
      backend_attributes, TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, nullptr, 0));

  return nullptr;
}

}  // extern "C"

}}}  // namespace triton::backend::openvino
