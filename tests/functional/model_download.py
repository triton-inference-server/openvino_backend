import os
import tarfile
from urllib.request import urlretrieve


def ir_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin",
        f"{repo}/{name}/1/model.bin",
    )
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml",
        f"{repo}/{name}/1/model.xml",
    )


def tflite_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    urlretrieve(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz",
        f"{repo}/{name}/tflite.tgz",
    )
    file = tarfile.open(f"{repo}/{name}/tflite.tgz")
    file.extractall(f"{repo}/{name}/1")
    file.close()
    os.rename(
        f"{repo}/{name}/1/inception_resnet_v2.tflite", f"{repo}/{name}/1/model.tflite"
    )


def onnx_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    urlretrieve(
        "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx",
        f"{repo}/{name}/1/model.onnx",
    )


def dynamic_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin",
        f"{repo}/{name}/1/model.bin",
    )
    urlretrieve(
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml",
        f"{repo}/{name}/1/model.xml",
    )


def paddle_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    urlretrieve(
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar",
        f"{repo}/{name}/paddle.tar",
    )
    file = tarfile.open(f"{repo}/{name}/paddle.tar")
    file.extractall(f"{repo}/{name}/1")
    file.close()
    os.rename(
        f"{repo}/{name}/1/MobileNetV3_large_x1_0_infer/inference.pdmodel",
        f"{repo}/{name}/1/model.pdmodel",
    )
    os.rename(
        f"{repo}/{name}/1/MobileNetV3_large_x1_0_infer/inference.pdiparams",
        f"{repo}/{name}/1/model.pdiparams",
    )


def saved_model(name, repo):
    os.makedirs(f"{repo}/{name}/1/model.saved_model")
    urlretrieve(
        "https://www.kaggle.com/api/v1/models/tensorflow/resnet-50/tensorFlow2/classification/1/download",
        f"{repo}/{name}/model.tar.gz",
    )
    file = tarfile.open(f"{repo}/{name}/model.tar.gz")
    file.extractall(f"{repo}/{name}/1/model.saved_model")
    file.close()
