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
+ openvino2022-device-for-mqtt.py: push the inference result of OpenVINO>=2022.1 to EdgeX by MQTT

### references
+ https://docs.ultralytics.com/quick-start/
+ https://github.com/ultralytics/yolov5/releases/tag/v6.1
+ https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
