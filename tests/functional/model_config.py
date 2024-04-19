MODEL_CONFIG = {
        "ir": {
            "shape": (1, 3, 224, 224),
            "input_name": "0",
            "output_name": "1463",
            "expected_shape": (1,1000)
            },
        "tflite": {
            "shape": (1, 299, 299, 3),
            "input_name": "input",
            "output_name": "InceptionResnetV2/AuxLogits/Logits/BiasAdd",
            "expected_shape": (1,1001)
            },
        "onnx": {
            "shape": (1, 3, 224, 224),
            "input_name": "gpu_0/data_0",
            "output_name": "gpu_0/softmax_1",
            "expected_shape": (1,1000)
            }
}
