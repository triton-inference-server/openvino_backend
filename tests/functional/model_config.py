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
