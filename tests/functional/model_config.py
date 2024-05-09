MODEL_CONFIG = {
        "ir": {
            "shape": [(1, 3, 224, 224)],
            "input_name": "0",
            "output_name": "1463",
            "expected_shape": [(1,1000)]
            },
        "tflite": {
            "shape": [(1, 299, 299, 3)],
            "input_name": "input",
            "output_name": "InceptionResnetV2/AuxLogits/Logits/BiasAdd",
            "expected_shape": [(1,1001)]
            },
        "onnx": {
            "shape": [(1, 3, 224, 224)],
            "input_name": "gpu_0/data_0",
            "output_name": "gpu_0/softmax_1",
            "expected_shape": [(1,1000)]
            },
        "paddle": {
            "shape": [(1, 3, 224, 224)],
            "input_name": "inputs",
            "output_name": "save_infer_model/scale_0.tmp_1",
            "expected_shape": [(1,1000)]
            },
        "pb": {
            "shape": [(1, 224, 224, 3)],
            "input_name": "input_1",
            "output_name": "activation_49",
            "expected_shape": [(1,1001)]
            },
        "dynamic": {
            "shape": [
                (1, 3, 224, 224),
                (2, 3, 224, 224),
                (3, 3, 224, 224),
                (1, 3, 200, 200),
                (1, 3, 300, 300)
                ],
            "input_name": "data",
            "output_name": "detection_out",
            "expected_shape": [(1,1,200,7), (1,1,400,7), (1,1,600,7)]
            }
}
