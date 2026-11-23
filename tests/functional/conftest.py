# Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
from model_config import CPU, GPU, INVALID_CONFIG_MODELS, MODEL_CONFIG, models
from tritonclient.grpc import service_pb2

CONTAINER_NAME = "openvino_backend_pytest"
ALL_MODELS = {**MODEL_CONFIG, **INVALID_CONFIG_MODELS}


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
    for model in models + list(INVALID_CONFIG_MODELS):
        if not os.path.exists(f"{cache}/{model}"):
            ALL_MODELS[model]["fetch"](model, cache)
    yield cache

    if dir is not None:
        dir.cleanup()


@pytest.fixture(scope="class", params=[CPU, pytest.param(GPU, marks=pytest.mark.gpu)])
def model_repository(model_cache, request):
    dir = tempfile.TemporaryDirectory()

    repo = dir.name

    # A test class can serve its own set of models via `repository_models`.
    for model in getattr(request.cls, "repository_models", models):
        gpu = GPU in ALL_MODELS[model]["configurations"] and request.param == GPU
        setup_model(model_cache, repo, model, gpu)

    yield repo

    dir.cleanup()
