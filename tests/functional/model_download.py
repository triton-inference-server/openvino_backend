import shutil
import os
import subprocess

def ir_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    cmd = ("curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin " 
                "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml " 
               f"-o {repo}/{name}/1/model.bin -o {repo}/{name}/1/model.xml")
    subprocess.run(cmd, shell=True)

def tflite_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    subprocess.run(f"wget -P {repo}/tflite/1 https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz 2> /dev/null", shell=True)
    subprocess.run(f"tar xzf {repo}/tflite/1/inception_resnet_v2_2018_04_27.tgz -C {repo}/{name}/1", shell=True)
    subprocess.run(f"mv {repo}/{name}/1/inception_resnet_v2.tflite {repo}/{name}/1/model.tflite", shell=True)

def onnx_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    subprocess.run(f"curl --fail -L --create-dir https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o {repo}/{name}/1/model.onnx", shell=True)

def dynamic_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    subprocess.run(f"wget -P {repo}/{name}/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin", shell=True)
    subprocess.run(f"wget -P {repo}/{name}/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml", shell=True)
    subprocess.run(f"mv {repo}/{name}/1/face-detection-retail-0004.bin {repo}/{name}/1/model.bin", shell=True)
    subprocess.run(f"mv {repo}/{name}/1/face-detection-retail-0004.xml {repo}/{name}/1/model.xml", shell=True)

def paddle_model(name, repo):
    os.makedirs(f"{repo}/{name}/1")
    subprocess.run(f"wget -P {repo}/{name}/1 https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar 2> /dev/null", shell=True)
    subprocess.run(f"tar xf {repo}/{name}/1/MobileNetV3_large_x1_0_infer.tar -C {repo}/{name}/1", shell=True)
    subprocess.run(f"mv {repo}/{name}/1/MobileNetV3_large_x1_0_infer/inference.pdmodel {repo}/{name}/1/model.pdmodel", shell=True)
    subprocess.run(f"mv {repo}/{name}/1/MobileNetV3_large_x1_0_infer/inference.pdiparams {repo}/{name}/1/model.pdiparams", shell=True)

def saved_model(name, repo):
    os.makedirs(f"{repo}/{name}/1/model.saved_model")
    subprocess.run(f"curl -L -o {repo}/{name}/1/model.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/resnet-50/tensorFlow2/classification/1/download", shell=True)
    subprocess.run(f"tar xzf {repo}/{name}/1/model.tar.gz -C {repo}/{name}/1/model.saved_model", shell=True)
