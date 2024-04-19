CONTAINER_NAME?="ovbackend_tests"

.PHONY: download
download:
	mkdir model_repository/ir/1 -p
	curl https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -o model_repository/ir/1/model.bin -o model_repository/ir/1/model.xml

	mkdir model_repository/tflite/1 -p
	wget -P model_repository/tflite/1 https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
	tar xzf model_repository/tflite/1/inception_resnet_v2_2018_04_27.tgz -C model_repository/tflite/1

	mkdir model_repository/onnx/1 -p
	curl --fail -L --create-dir https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o model_repository/onnx/1/model.onnx

	mkdir model_repository/dynamic/1 -p
	wget -P model_repository/dynamic/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin
	wget -P model_repository/dynamic/1 https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
	mv model_repository/dynamic/1/face-detection-retail-0004.bin model_repository/dynamic/1/model.bin
	mv model_repository/dynamic/1/face-detection-retail-0004.xml model_repository/dynamic/1/model.xml

.PHONY: run
run:
	@docker run -p 8111:8001 --rm -d -v ${PWD}/model_repository:/model_repository --name=$(CONTAINER_NAME) tritonserver:latest bin/tritonserver --model-repository /model_repository 
	@sleep 3
	-pytest functional/
	@docker stop $(CONTAINER_NAME)





