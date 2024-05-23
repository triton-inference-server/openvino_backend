import errno
import os
import shutil
import socket
import subprocess
import tempfile

import pytest
from model_config import CPU, GPU, MODEL_CONFIG, models

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
    image_name = request.config.getoption("--image")
    image_name = image_name if image_name is not None else "tritonserver:latest"
    gpu = '--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* )' if request.config.getoption("--gpu") else ""
    subprocess.run(
        f'docker run -p {port}:8001 -d -v {model_repository}:/model_repository --name={CONTAINER_NAME} {gpu}  {image_name} bin/tritonserver --model-repository /model_repository',
        capture_output=True,
        shell=True,
    )
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
def model_cache(request):
    input_dir = request.config.getoption('--model-cache')
    dir = None
    if input_dir == "":
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
