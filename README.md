<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

The Triton backend for the
[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html). You
can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend).  Ask
questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues). The backend
is designed to run models in Intermediate Representation (IR). See [here](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html) for instruction to convert a model to IR format. The backend is implemented using openVINO C++ API. Auto completion of the model config is not supported in the backend and complete `config.pbtxt` must be provided with the model.


Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
main Triton [issues page](https://github.com/triton-inference-server/server/issues).

**Note:** OpenVINO backend is in early stage of developement and is not yet ready for general use.

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

## Advanced

### Parameters

The configuration is done through the `config.pbtxt` file. The parameters and their description are as follows.

* `CPU_EXTENSION_PATH`: Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
* `CPU_THREADS_NUM`: Number of threads to use for inference on the CPU. Should be a non-negative number.
* `ENFORCE_BF16`: Enforcing of floating point operations execution in bfloat16 precision on platforms with native bfloat16 support. Possible values are `YES` or `NO`.
* `CPU_BIND_THREAD`: Enable threads->cores (`YES`, default), threads->(NUMA)nodes (`NUMA`) or completely disable (`NO`) CPU threads pinning for CPU-involved inference.
* `CPU_THROUGHPUT_STREAMS`: Number of streams to use for inference on the CPU. Default value is determined automatically for a device. Please note that although the automatic selection usually provides a reasonable performance, it still may be non-optimal for some cases, especially for very small networks. Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency estimations the number of streams should be set to 1.
