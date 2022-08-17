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
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_OPENVINO_VERSION=2021.2.200 -DTRITON_BUILD_CONTAINER_VERSION=20.12 ..
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

* `CPU_EXTENSION_PATH`: Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
* `INFERENCE_NUM_THREADS`: Number of threads to use for inference on the CPU. Should be a non-negative number.
* `COMPILATION_NUM_THREADS`: Number of threads to use for model loading/compilation. Should be a non-negative number.
* `HINT_BF16`: Hint of floating point operations execution in bfloat16 precision on platforms with native bfloat16 support. Possible value is `YES`.
* `NUM_STREAMS`: Enable threads->cores (`YES`, default) or threads->(NUMA)nodes (`NUMA`) CPU threads pinning for CPU-involved inference.
* `SKIP_OV_DYNAMIC_BATCHSIZE`: The topology of some models do not support openVINO dynamic batch sizes. Set the value of this parameter to `YES`, in order
to skip the dynamic batch sizes in backend.
* `ENABLE_BATCH_PADDING`: By default an error will be generated if backend receives a request with batch size less than max_batch_size specified in the configuration. This error can be avoided at a cost of performance by specifying `ENABLE_BATCH_PADDING` parameter as `YES`.
* `RESHAPE_IO_LAYERS`: By setting this parameter as `YES`, the IO layers are reshaped to the dimensions provided in
model configuration. By default, the dimensions in the model is used.

The section of model config file specifying these parameters will look like:

```
.
.
.
parameters: {
key: "NUM_STREAMS"
value: {
string_value:"NUMA"
}
}
parameters: {
key: "INFERENCE_NUM_THREADS"
value: {
string_value:"5"
}
}
.
.
.

```

## Known Issues

* Not all models support dynamic batch sizes.

* As of now, the Openvino backend does not support variable shaped tensors. However, the dynamic batch sizes in the model are supported. See `SKIP_OV_DYNAMIC_BATCHSIZE` and `ENABLE_BATCH_PADDING` parameters for more details.

* Openvino does not support CPU execution for FP16.
