import subprocess
import pytest
import socket
import os
import tempfile
import shutil
from model_config import MODEL_CONFIG, CPU, GPU, models
CONTAINER_NAME="openvino_backend_pytest"


@pytest.fixture(scope="class")
def triton_server(model_repository):
    port = 0
    try:
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
    except socket.error as e:
        if e.errno != errno.EADDRINUSE:
            raise Exception(f"Not expected exception found in port manager {self.name}: {e}")
    image_name = os.environ.get("TIS_IMAGE_NAME")
    image_name = image_name if image_name != None else "tritonserver:latest"
    subprocess.run(f"docker run -p {port}:8001 -d -v {model_repository}:/model_repository --name={CONTAINER_NAME} --device /dev/dri --group-add=$(stat -c \"%g\" /dev/dri/render* ) {image_name} bin/tritonserver --model-repository /model_repository", capture_output=True, shell=True)
    subprocess.run(["sleep", "3"])
    subprocess.run(["docker", "logs", CONTAINER_NAME])
    yield port
    subprocess.run(["docker", "stop", CONTAINER_NAME])
    subprocess.run(["docker", "remove", CONTAINER_NAME])


def copy_config(repo, name, gpu=False):
    gpu_suffix = "_gpu" if gpu else ""
    shutil.copy(f"configs/{name}{gpu_suffix}.pbtxt", f"{repo}/{name}/config.pbtxt")

def setup_model(cache, repo, name, gpu=False):
    shutil.copytree(f"{cache}/{name}", f"{repo}/{name}") 
    copy_config(repo, name, gpu)


@pytest.fixture(scope="session")
def model_cache():
    dir = tempfile.TemporaryDirectory()
    cache = dir.name
    for model in models:
        MODEL_CONFIG[model]['fetch'](model, cache)
    yield cache

    dir.cleanup()

@pytest.fixture(scope="class", params=[CPU, pytest.param(GPU, marks=pytest.mark.gpu)])
def model_repository(model_cache, request):
    dir = tempfile.TemporaryDirectory()

    repo = dir.name

    for model in models:
        gpu = False
        if GPU in MODEL_CONFIG[model]["configurations"] and request.param == GPU:
            gpu = True
        setup_model(model_cache, repo, model, gpu)

    yield repo

    dir.cleanup()
