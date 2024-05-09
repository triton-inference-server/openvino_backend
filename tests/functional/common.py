import numpy as np
import grpc
import os
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2_grpc
from tritonclient.utils import np_to_triton_dtype

def prepare_inputs(input_name, shape, dtype):
    inputs = []
    inputs.append(grpcclient.InferInput(input_name, shape, np_to_triton_dtype(dtype)))

    inputs[0].set_data_from_numpy(np.ones(shape, dtype=dtype))
    return inputs

def prepare_grpc_stub(port):
    channel = grpc.insecure_channel(f"localhost:{port}")
    return service_pb2_grpc.GRPCInferenceServiceStub(channel)

def prepare_triton_client(port):
    return grpcclient.InferenceServerClient(
        url=f"localhost:{port}",
        verbose=os.environ.get("LOG_LEVEL")=="DEBUG")
