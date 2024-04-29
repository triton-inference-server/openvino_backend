import numpy as np
import grpc
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2_grpc

def prepare_inputs(input_name, shape):
    inputs = []
    inputs.append(grpcclient.InferInput(input_name, shape, "FP32"))

    inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
    return inputs

def prepare_grpc_stub(port):
    channel = grpc.insecure_channel(f"localhost:{port}")
    return service_pb2_grpc.GRPCInferenceServiceStub(channel)

def prepare_triton_client(port):
    return grpcclient.InferenceServerClient(
        url=f"localhost:{port}",
        verbose=False)
