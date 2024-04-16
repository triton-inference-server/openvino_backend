<!--
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# OpenVINO Backend

**Note: OpenVINO backend is beta quality. As a result you may
encounter performance and functional issues that will be resolved in
future releases.**

The Triton backend for the
[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html). You
can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend).  Ask
questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues). The backend
is designed to run models in Intermediate Representation (IR). See [here](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) for instruction to convert a model to IR format. The backend is implemented using openVINO C++ API. Auto completion of the model config is not supported in the backend and complete `config.pbtxt` must be provided with the model.

## Supported Devices
OpenVINO backend currently supports inference only on Intel CPU devices using [OpenVINO CPU plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CPU.html). Note the CPU plugin does not support
iGPU.

## Build the OpenVINO Backend

Cmake 3.17 or higher is required. First install the required dependencies.

```
$ apt-get install patchelf rapidjson-dev python3-dev
```

Follow the steps below to build the backend shared library.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_OPENVINO_VERSION=2024.0.0 -DTRITON_BUILD_CONTAINER_VERSION=24.03 ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Using the OpenVINO Backend

### Parameters

Configuration of OpenVINO for a model is done through the Parameters section of the model's 'config.pbtxt' file. The parameters and their description are as follows.

* `PERFORMANCE_HINT`: Presetting performance tuning options. Accepted values `LATENCY` for low concurrency use case and `THROUGHPUT` for high concurrency scenarios.
* `CPU_EXTENSION_PATH`: Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
* `INFERENCE_NUM_THREADS`: Maximum number of threads that can be used for inference tasks. Should be a non-negative number. Default is equal to number of cores.
* `COMPILATION_NUM_THREADS`: Maximum number of threads that can be used for compilation tasks. Should be a non-negative number.
* `HINT_BF16`: Hint for device to use bfloat16 precision for inference. Possible value is `YES`.
* `NUM_STREAMS`: The number of executor logical partitions. Set the value to `AUTO` to creates bare minimum of streams to improve the performance, or set the value to `NUMA` to creates as many streams as needed to accommodate NUMA and avoid associated penalties. Set a numerical value to set explicit number of streams.
* `SKIP_OV_DYNAMIC_BATCHSIZE`: The topology of some models do not support openVINO dynamic batch sizes. Set the value of this parameter to `YES`, in order
to skip the dynamic batch sizes in backend.
* `ENABLE_BATCH_PADDING`: By default an error will be generated if backend receives a request with batch size less than max_batch_size specified in the configuration. This error can be avoided at a cost of performance by specifying `ENABLE_BATCH_PADDING` parameter as `YES`.
* `RESHAPE_IO_LAYERS`: By setting this parameter as `YES`, the IO layers are reshaped to the dimensions provided in
model configuration. By default, the dimensions in the model is used.



## Auto-Complete Model Configuration

Assuming Triton was not started with `--disable-auto-complete-config` command line
option, the OpenVINO Backend makes use of the model configuration available in
OpenVINO models to populate the required fields in the model's "config.pbtxt".
You can learn more about Triton's support for auto-completing model
configuration from
[here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration).

However, not all OpenVINO models carry sufficient configuration information to
auto-complete the model's "config.pbtxt". As a result, a partial "config.pbtxt"
could still be required for some models.

OpenVINO backend can complete the following fields in model configuration:

### max_batch_size

Auto-completing max_batch_size follows the following rules:

- The model has included sufficient layout information.
- Autocomplete has determined the model is capable of batching requests.
- `max_batch_size` is 0 in the model configuration or max_batch_size is omitted
from the model configuration.

If the above two rules are met, max_batch_size is set to
default-max-batch-size. Otherwise max_batch_size is set as 0.

### Inputs and Outputs

The OpenVINO Backend is able to fill in the `name`, `data_type`, and `dims` provided this
information is available in the model.

Autocompleting inputs/outputs follows the following rules:
- If `inputs` and/or `outputs` is empty or undefined in the model
configuration, the missing inputs and/or outputs will be autocompleted.
- Auto-complete will skip over any defined/filled-in inputs and outputs.

### Dynamic Batching

If `max_batch_size > 1`, after auto-completing `max_batch_size`, and no
[`dynamic_batching`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)
and
[`sequence_batching`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#sequence-batcher)
is provided, then `dynamic_batching` will be enabled with default settings.


## Examples of the "config.pbtxt" file sections depending on the use case

### Latency mode

Latency mode with low concurrency on the client side. Recommended for performance optimization with low number of parallel clients.
```
parameters: [
{
   key: "NUM_STREAMS"
   value: {
     string_value: "1"
   }
},
{
   key: "PERFORMANCE_HINT"
   value: {
     string_value: "LATENCY"
   }
}
]
```

### Throughput mode

Throughput mode with high concurrency on the client side. Recommended for throughput optimization with high number of parallel clients.
Number of streams should be lower or equal to number of parallel clients and lower of equal to the number of CPU cores.
For example, with ~20 clients on the host with 12 CPU cores, the config could be like:
```
instance_group [
    {
      count: 12
      kind: KIND_CPU
    }
  ]
parameters: [
{
   key: "NUM_STREAMS"
   value: {
     string_value: "12"
   }
}
]
```

### Loading non default model format

When loading model with the non default format of Intermediate Representation and the name model.xml, use and extra parameter "default_model_filename".
For example, using TensorFlow saved_model format use:
```
default_model_filename: "model.saved_model"
```

and copy the model to the subfolder called "model.saved_model"
```
model_repository/
└── model
    ├── 1
    │   └── model.saved_model
    │       ├── saved_model.pb
    │       └── variables
    └── config.pbtxt

```
Other allowed values are `model.pdmodel` or `model.onnx`.
### Reshaping models

Following section shows how to use OpenVINO dynamic shapes. `-1` denotes dimension accepting any value on input. In this case
while model originally accepted input with layout `NCHW` and shape `(1,3,224,224)`, now it accepts any batch size and resolution.

*Note*: If the model is originally exported with dynamic shapes, there is no need to manually specify dynamic shapes in config.

*Note*: Some models might not support a shape with an arbitrary dimension size. An error will be raised during the model initialization when the shaped in the config.pbtxt is not possible. Clients will also receive an error if the request includes unsupported input shapes.
```
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ -1, 3, -1, -1]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1, 1001]
  }
]
parameters: {
key: "RESHAPE_IO_LAYERS"
value: {
string_value:"yes"
}
}
```

## Known Issues

* Models with the scalar on the input (shape without any dimension are not supported)
* Reshaping using [dimension ranges](https://docs.openvino.ai/2023.3/ovms_docs_dynamic_shape_dynamic_model.html) is not supported.
