import errno
import os
import shutil
import socket
import subprocess
import tempfile
import time

import common
import pytest
import requests
from model_config import CPU, GPU, MODEL_CONFIG, models
from tritonclient.grpc import service_pb2

CONTAINER_NAME = "openvino_backend_pytest"


@pytest.fixture(scope="class")
def triton_server(model_repository, request):
    port = 0
    try:
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
    except socket.error as e:
        if e.errno != errno.EADDRINUSE:
            raise Exception(f"Not expected exception found in port manager: {e.errno}")
    container_name = f"{CONTAINER_NAME}{port}"
    image_name = request.config.getoption("--image")
    image_name = image_name if image_name is not None else "tritonserver:latest"
    gpu = (
        '--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* )'
        if request.config.getoption("--gpu")
        else ""
    )
    print(
        f"Starting triton server {image_name} with name {container_name}, additional parameters: {gpu}"
    )
    subprocess.run(
        f"docker run -p {port}:8001 -d -v {model_repository}:/model_repository --name={container_name} {gpu}  {image_name} bin/tritonserver --model-repository /model_repository --exit-on-error 0",
        capture_output=True,
        shell=True,
    )
    print("Waiting for the server to initialize")
    grpc_stub = common.prepare_grpc_stub(port)
    request = service_pb2.ServerReadyRequest()
    ready = 0
    timeout = 10
    now = time.time()
    while ready == 0:
        if time.time() - now > timeout:
            break
        try:
            response = grpc_stub.ServerReady(request)
            ready = response.ready
        except Exception:
            pass
    subprocess.run(["docker", "logs", container_name])
    yield port
    print(f"Stopping container {container_name}")
    subprocess.run(["docker", "stop", container_name])
    print(f"Cleaning up {container_name}")
    subprocess.run(
        f"docker container rm {container_name}", shell=True, capture_output=True
    )


def copy_config(repo, name, gpu=False):
    gpu_suffix = "_gpu" if gpu else ""
    shutil.copy(f"configs/{name}{gpu_suffix}.pbtxt", f"{repo}/{name}/config.pbtxt")


def setup_model(cache, repo, name, gpu=False):
    shutil.copytree(f"{cache}/{name}", f"{repo}/{name}")
    copy_config(repo, name, gpu)


@pytest.fixture(scope="session")
def model_cache(request):
    input_dir = request.config.getoption("--model-cache")
    dir = None
    if input_dir is None:
        dir = tempfile.TemporaryDirectory()
        cache = dir.name
    else:
        cache = input_dir
    if os.listdir(cache) == []:
        for model in models:
            MODEL_CONFIG[model]["fetch"](model, cache)
    yield cache

    if dir is not None:
        dir.cleanup()


@pytest.fixture(scope="class", params=[CPU, pytest.param(GPU, marks=pytest.mark.gpu)])
def model_repository(model_cache, request):
    dir = tempfile.TemporaryDirectory()

    repo = dir.name

    for model in models:
        gpu = GPU in MODEL_CONFIG[model]["configurations"] and request.param == GPU
        setup_model(model_cache, repo, model, gpu)

    yield repo

    dir.cleanup()
