# YOLOv5 Inference Demo by OpenCV and OpenVINO
Do the YOLOv5 model inference by OpenCV/OpenVINO based on onnx model format
### Installation
Clone the repository
$ git clone https://github.com/ultralytics/yolov5.git
Enter the repository root directory

$ cd yolov5
Install the required packages from your cloned repository root directory

$ pip install -r requirements.txt

### Export the YOLOv5 model to onnx model
$ python export.py --weights yolov5s.pt --include onnx

### demo code
+ infer_by_opencv.py: do the inference by the OpenCV DNN module
+ infer_by_openvino.py: do the inference by the OpenVINO<=2021.4.2
+ infer_by_openvino2022.py: do the inference by the OpenVINO>=2022.1
+ test_env.cpp: test the VS2019 development environment based on OpenVINO>=2022.1 and OpenCV, C++ version
+ yolov5_ov2022_cam.cpp: sample code about do the yolov5 inference by USB camera. OpenVINO>=2022.1 C++ version
+ yolov5_ov2022_image.cpp:sample code about do the yolov5 inference on one image. OpenVINO>=2022.1 C++ version
+ infer_with_openvino_preprocess.py: sample code about do the yolov5 inference with OpenVINO preprocessing API. OpenVINO>=2022.1 C++ version
+ openvino2022-device-for-mqtt.py: push the inference result of OpenVINO>=2022.1 to EdgeX by MQTT
+ ov_cvmart_sample.ipynb: cvmart newbie task OpenVINO>=2022.1 sample [基于YOLOv5的新手任务](https://www.cvmart.net/document)
+ ji.py: cvmart newbie task, auto test script.[基于YOLOv5的新手任务,编写测试脚本](https://www.cvmart.net/document)

### references
+ https://docs.ultralytics.com/quick-start/
+ https://github.com/ultralytics/yolov5/releases/tag/v6.1
+ https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
+ https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109
