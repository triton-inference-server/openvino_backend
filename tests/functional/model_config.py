# Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
from model_download import *

CPU = "cpu"
GPU = "gpu"

MODEL_CONFIG = {
    "ir": {
        "enable": True,
        "fetch": ir_model,
        "shape": [(1, 3, 224, 224)],
        "configurations": [CPU, GPU],
        "input_name": "0",
        "output_name": "1463",
        "dtype": np.float32,
        "expected_shape": [(1, 1000)],
    },
    "ir_precision_f32": {
        "enable": True,
        "fetch": ir_model,
        "shape": [(1, 3, 224, 224)],
        "configurations": [CPU],
        "input_name": "0",
        "output_name": "1463",
        "dtype": np.float32,
        "expected_shape": [(1, 1000)],
    },
    "tflite": {
        "enable": True,
        "fetch": tflite_model,
        "shape": [(1, 299, 299, 3)],
        "configurations": [CPU],
        "input_name": "input",
        "output_name": "InceptionResnetV2/AuxLogits/Logits/BiasAdd",
        "dtype": np.float32,
        "expected_shape": [(1, 1001)],
    },
    "onnx": {
        "enable": True,
        "fetch": onnx_model,
        "shape": [(1, 3, 224, 224)],
        "configurations": [CPU, GPU],
        "input_name": "gpu_0/data_0",
        "output_name": "gpu_0/softmax_1",
        "dtype": np.float32,
        "expected_shape": [(1, 1000)],
    },
    "paddle": {
        "enable": True,
        "fetch": paddle_model,
        "shape": [(1, 3, 224, 224)],
        "configurations": [CPU, GPU],
        "input_name": "inputs",
        "output_name": "save_infer_model/scale_0.tmp_1",
        "dtype": np.float32,
        "expected_shape": [(1, 1000)],
    },
    "saved_model": {
        "enable": True,
        "fetch": saved_model,
        "shape": [(1, 224, 224, 3)],
        "configurations": [CPU],
        "input_name": "input_1",
        "output_name": "activation_49",
        "dtype": np.float32,
        "expected_shape": [(1, 1001)],
    },
    "dynamic": {
        "enable": True,
        "fetch": dynamic_model,
        "configurations": [
            CPU
        ],  # [GPU] PriorBoxClustered op is not supported in GPU plugin yet.
        "shape": [
            (1, 3, 224, 224),
            (2, 3, 224, 224),
            (3, 3, 224, 224),
            (1, 3, 200, 200),
            (1, 3, 300, 300),
        ],
        "input_name": "data",
        "output_name": "detection_out",
        "dtype": np.float32,
        "expected_shape": [(1, 1, 200, 7), (1, 1, 400, 7), (1, 1, 600, 7)],
    },
}
models = list(filter(lambda it: MODEL_CONFIG[it]["enable"], MODEL_CONFIG.keys()))

# Models whose config must be rejected at load time. They are served by their
# own server instance (see TestInvalidConfig), so they never block the
# readiness of the models above.
INVALID_CONFIG_MODELS = {
    "ir_precision_invalid": {
        "fetch": ir_model,
        "configurations": [CPU],
        "expected_error": "INFERENCE_PRECISION_HINT",
    },
}
