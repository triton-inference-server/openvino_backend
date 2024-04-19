import numpy as np
import grpc
import pytest

from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
import tritonclient.grpc as grpcclient
from model_config import MODEL_CONFIG
PORT=8111


@pytest.mark.parametrize("model", MODEL_CONFIG.keys())
def test_inference(model):
    triton_client = grpcclient.InferenceServerClient(
        url=f"localhost:{PORT}",
        verbose=False)

    config = MODEL_CONFIG[model]

    inputs = []
    inputs.append(grpcclient.InferInput(config["input_name"], config["shape"], "FP32"))

    inputs[0].set_data_from_numpy(np.ones(config["shape"], dtype=np.float32))
    results = triton_client.infer(
        model_name= model,
        inputs=inputs)
    output = results.as_numpy(config["output_name"])
    assert output.shape == config["expected_shape"]

def test_dynamic_shape_inference():
    triton_client = grpcclient.InferenceServerClient(
        url=f"localhost:{PORT}",
        verbose=False)


    inputs = []
    SHAPES = [(1, 3, 300, 300)]
    for shape in SHAPES:
        inputs.append(grpcclient.InferInput("data", shape, "FP32"))

        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        results = triton_client.infer(
            model_name= "dynamic",
            inputs=inputs)
        output = results.as_numpy("detection_out")
        assert output.shape == (1, 1, 200, 7)

@pytest.mark.parametrize("model", MODEL_CONFIG.keys())
def test_model_ready(model):
    channel = grpc.insecure_channel(f"localhost:{PORT}")
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    request = service_pb2.ModelReadyRequest(name=model)
    response = grpc_stub.ModelReady(request)
    assert response.ready == True

@pytest.mark.parametrize("model", MODEL_CONFIG.keys())
def test_model_metadata(model):
    channel = grpc.insecure_channel(f"localhost:{PORT}")
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    request = service_pb2.ModelMetadataRequest(name=model)
    response = grpc_stub.ModelMetadata(request)
    assert response is not None

