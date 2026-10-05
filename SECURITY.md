<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Security Policy

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for a suspected vulnerability.
Report it through one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program (preferred):**
   <https://www.nvidia.com/en-us/security/>
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt the
   report with the [NVIDIA PGP key](https://www.nvidia.com/en-us/security/pgp-key).
3. **GitHub Private Vulnerability Reporting (where enabled):** use the "Report a vulnerability"
   button on the Security tab of this repository.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   memory corruption)
3. Steps to reproduce
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, assesses severity, coordinates a fix and
publishes a security bulletin where appropriate. See
<https://www.nvidia.com/en-us/security/> for past bulletins and notices.

Vulnerabilities in the OpenVINO runtime itself should also be reported to the
OpenVINO project. This backend is co-maintained by NVIDIA and Intel.

## Security Architecture and Context

**Project:** Triton OpenVINO Backend, a C++ plugin
(`libtriton_openvino.so`) for the Triton Inference Server that runs models
through the OpenVINO C++ API.

**Classification:** Library / plugin. It is loaded in-process by the Triton
server through the `TRITONBACKEND_*` API and has no network listener, command
line interface or authentication logic of its own.

**Repository Exposure Classification:** Public. Basis: the repository is
publicly visible on GitHub.

**Service Exposure Classification:** Not determined (low confidence). Basis:
the backend is a component whose exposure depends on how the hosting Triton
server is deployed; no regulatory scope can be derived from this repository
alone.

**Primary security responsibility:** safely load model artifacts that were
placed in a Triton model repository, validate tensor shapes and sizes taken from
inference requests against the model configuration, and execute inference on
the configured OpenVINO device without corrupting host memory. The device is the
CPU by default; the `TARGET_DEVICE` parameter can select GPU, NPU or a virtual
device such as AUTO, MULTI or HETERO.

**Key interfaces and boundaries:**

- **Model repository (file system):** `src/openvino.cc` resolves
  `<repository>/<version>/<artifact>` (default `model.xml`) and passes it to
  `ov::Core::read_model`. OpenVINO parses IR, ONNX, TensorFlow SavedModel,
  TensorFlow Lite and PaddlePaddle formats from this path.
- **Model configuration (`config.pbtxt`):** `parameters` such as
  `INFERENCE_NUM_THREADS`, `NUM_STREAMS`, `PERFORMANCE_HINT`,
  `RESHAPE_IO_LAYERS` and `SKIP_OV_DYNAMIC_BATCHSIZE`, and input/output
  `reshape` entries, are parsed by `ParseParameters` and `ParseShape`.
- **Inference requests:** input tensors reach the backend through the Triton
  core (`BackendInputCollector`) and are copied into OpenVINO tensors.
- **Backend configuration:** the `cmdline` block from
  `TRITONBACKEND_BackendConfig` is parsed at backend initialization.
- **Build and packaging:** `tools/gen_openvino_dockerfile.py` and
  `Dockerfile.drivers` fetch and build OpenVINO from its upstream sources.

## Threat Model

1. **Malicious or corrupted model artifact:** an attacker who can write to the
   model repository supplies a crafted IR, ONNX, TFLite, SavedModel or Paddle
   file. The file is parsed by `ov::Core::read_model` in `ModelState::ReadModel`
   inside the Triton server process, so a parser flaw in OpenVINO can lead to
   memory corruption or code execution with the server's privileges.
2. **Untrusted model configuration values:** `config.pbtxt` parameters and
   `reshape` dimensions are converted to OpenVINO properties and shapes
   (`ParseParameterHelper`, `ParseShape`). Extreme thread, stream or dimension
   values can cause resource exhaustion or integer-overflow in size
   calculations.
3. **Malformed inference requests:** request tensors are copied into
   OpenVINO input tensors by `ModelInstanceState::SetInputTensors`, which
   `ProcessRequests` calls. A mismatch between the declared shape and the byte
   size supplied by the client could cause an out-of-bounds read or write. On
   the default path the backend compares the expected and received byte sizes
   and returns an error on a mismatch. When `ENABLE_BATCH_PADDING` is set, the
   padded path only logs a verbose message about a size difference and copies
   into a buffer sized to the input tensor, so operators should not rely on the
   size check for padded requests.
4. **Denial of service through inference load:** large or dynamically shaped
   batches, unbounded request concurrency and expensive models can exhaust CPU
   and memory on the host shared with other models.
5. **Supply-chain compromise of the build:** the generated Dockerfile clones
   OpenVINO by tag with submodules and installs Python build tools from package
   registries without pinned hashes for all components. A compromised upstream
   or mutable tag would affect the shipped binary.
6. **Information disclosure through error messages and logs:** error strings
   include model file paths and OpenVINO exception text, which may expose the
   repository layout to clients that can read server errors.

## Critical Security Assumptions

- **Model repository is trusted.** The backend does not sandbox or verify model
  files. Operators must restrict write access to the repository and obtain
  models only from trusted sources.
- **The deployer provides authentication, authorization, TLS and rate
  limiting.** This backend implements none of them, and Triton does not
  provide user identity or per-user authorization itself. Operators must
  configure the applicable server or gateway controls before requests reach the
  backend.
- **The OpenVINO runtime is trusted and kept up to date.** Model parsing is
  delegated to it; vulnerabilities in OpenVINO are inherited by this backend.
  When `TARGET_DEVICE` selects GPU or NPU, the corresponding device plugins and
  drivers are part of the trusted runtime.
- **Shape and size validation depends on a correct model configuration.**
  Operators must keep `config.pbtxt` consistent with the deployed model.
- **The server process runs with least privilege** and with resource limits
  (memory, CPU, file descriptors), because the backend runs in-process and does
  not isolate faults.
- **Build inputs are trusted.** Builds assume the upstream OpenVINO source and
  base images are authentic.

## Supported Versions

Security fixes are made on the `main` branch and the most recent Triton release
branch. Use the backend release that matches your Triton container version.
