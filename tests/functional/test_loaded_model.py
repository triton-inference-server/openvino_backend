import common
import pytest
from model_config import MODEL_CONFIG, models
from tritonclient.grpc import service_pb2


@pytest.mark.parametrize("model", models)
class TestLoadedModel:
    def test_inference(self, model, triton_server):
        triton_client = common.prepare_triton_client(triton_server)

        config = MODEL_CONFIG[model]

        for shape in config["shape"]:
            inputs = common.prepare_inputs(config["input_name"], shape, config["dtype"])
            results = triton_client.infer(model_name=model, inputs=inputs)
            output = results.as_numpy(config["output_name"])
            assert (
                output.shape in config["expected_shape"]
            ), 'Invalid shape returned from the server, expected {config["expected_shape"]} got {output.shape}'

    def test_model_ready(self, model, triton_server):
        grpc_stub = common.prepare_grpc_stub(triton_server)
        request = service_pb2.ModelReadyRequest(name=model)
        response = grpc_stub.ModelReady(request)
        assert (
            response.ready
        ), f"Invalid status reported from the server, expected True got {response.ready}"

    def test_model_metadata(self, model, triton_server):
        grpc_stub = common.prepare_grpc_stub(triton_server)
        request = service_pb2.ModelMetadataRequest(name=model)
        response = grpc_stub.ModelMetadata(request)
        assert response is not None, "No response was reposted from the server"
