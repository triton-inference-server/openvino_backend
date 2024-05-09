import subprocess
import pytest
import socket
import os
import tempfile
import shutil
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
            raise Exception("Not expected exception found in port manager {}: {}".format(self.name, e))
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

def ir_model(repo):
    os.makedirs(f"{repo}/ir/1")
    subprocess.run(f"curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o {repo}/ir/1/model.bin -o {repo}/ir/1/model.xml", shell=True)
    copy_config(repo, "ir")

def tflite_model(repo):
    os.makedirs(f"{repo}/tflite/1")
    subprocess.run(f"wget -P {repo}/tflite/1 https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz 2> /dev/null", shell=True)
    subprocess.run(f"tar xzf {repo}/tflite/1/inception_resnet_v2_2018_04_27.tgz -C {repo}/tflite/1", shell=True)
    subprocess.run(f"mv {repo}/tflite/1/inception_resnet_v2.tflite {repo}/tflite/1/model.tflite", shell=True)
    copy_config(repo, "tflite")

def onnx_model(repo):
    os.makedirs(f"{repo}/onnx/1")
    subprocess.run(f"curl --fail -L --create-dir https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o {repo}/onnx/1/model.onnx", shell=True)
    copy_config(repo, "onnx")

def dynamic_model(repo):
    os.makedirs(f"{repo}/dynamic/1")
    subprocess.run(f"wget -P {repo}/dynamic/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin", shell=True)
    subprocess.run(f"wget -P {repo}/dynamic/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml", shell=True)
    subprocess.run(f"mv {repo}/dynamic/1/face-detection-retail-0004.bin {repo}/dynamic/1/model.bin", shell=True)
    subprocess.run(f"mv {repo}/dynamic/1/face-detection-retail-0004.xml {repo}/dynamic/1/model.xml", shell=True)
    copy_config(repo, "dynamic")

def paddle_model(repo):
    os.makedirs(f"{repo}/paddle/1")
    subprocess.run(f"wget -P {repo}/paddle/1 https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar 2> /dev/null", shell=True)
    subprocess.run(f"tar xf {repo}/paddle/1/MobileNetV3_large_x1_0_infer.tar -C {repo}/paddle/1", shell=True)
    subprocess.run(f"mv {repo}/paddle/1/MobileNetV3_large_x1_0_infer/inference.pdmodel {repo}/paddle/1/model.pdmodel", shell=True)
    subprocess.run(f"mv {repo}/paddle/1/MobileNetV3_large_x1_0_infer/inference.pdiparams {repo}/paddle/1/model.pdiparams", shell=True)
    copy_config(repo, "paddle")

def pb_model(repo):
    os.makedirs(f"{repo}/pb/1/model.saved_model")
    subprocess.run(f"curl -L -o {repo}/pb/1/model.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/resnet-50/tensorFlow2/classification/1/download", shell=True)
    subprocess.run(f"tar xzf {repo}/pb/1/model.tar.gz -C {repo}/pb/1/model.saved_model", shell=True)
    copy_config(repo, "pb")

def setup_model(cache, repo, name, gpu=False):
    shutil.copytree(f"{cache}/{name}", f"{repo}/{name}") 
    copy_config(repo, name, gpu)


@pytest.fixture(scope="session")
def model_cache():
    dir = tempfile.TemporaryDirectory()
    cache = dir.name
    ir_model(cache)
    tflite_model(cache)
    onnx_model(cache)
    dynamic_model(cache)
    paddle_model(cache)
    pb_model(cache)
    yield cache

    dir.cleanup()

@pytest.fixture(scope="class", params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def model_repository(model_cache, request):
    dir = tempfile.TemporaryDirectory()

    repo = dir.name

    setup_model(model_cache, repo, "ir", request.param == "gpu")
    setup_model(model_cache, repo, "tflite")
    setup_model(model_cache, repo, "onnx", request.param == "gpu")
    setup_model(model_cache, repo, "dynamic") # [GPU] PriorBoxClustered op is not supported in GPU plugin yet.
    setup_model(model_cache, repo, "paddle", request.param == "gpu")
    setup_model(model_cache, repo, "pb")

    yield repo

    dir.cleanup()
