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
+ infer_with_openvino_preprocess.py: sample code about do the yolov5 inference in ASYNC mode with OpenVINO preprocessing API. OpenVINO>=2022.1 Python version
+ infer_with_openvino_preprocess_sync.py: sample code about do the yolov5 inference in SYNC mode with OpenVINO preprocessing API. OpenVINO>=2022.1 Python version
+ preprocessing_with_saving_to_IR.py: sample code about export the IR model with preprocessing
+ openvino2022-device-for-mqtt.py: push the inference result of OpenVINO>=2022.1 to EdgeX by MQTT
+ ov_cvmart_sample.ipynb: cvmart newbie task OpenVINO>=2022.1 sample [基于YOLOv5的新手任务](https://www.cvmart.net/document)
+ yolov5_ov2022_sync_dGPU.py: do the yolov5 Sync inference on intel discreate GPU by OpenVINO2022.2, pls "pip install -U yolort" firstly, refer to https://github.com/zhiqwang/yolov5-rt-stack
+ yolov5_ov2022_async_dGPU.py: do the yolov5 Async inference on intel discreate GPU by OpenVINO2022.2, pls "pip install -U yolort" firstly, refer to https://github.com/zhiqwang/yolov5-rt-stack
+ ji.py: cvmart newbie task, auto test script.[基于YOLOv5的新手任务,编写测试脚本](https://www.cvmart.net/document)
+ yolov5seg_ov2022_sync_dGPU.py: yolov5 instance segmentation sample code by >=OpenVINO2022.2 on Intel A770
+ yolov5_async_infer_queue.py: YOLOv5 Async Infer based on OpenVINO AsyncInferQueue
+ yolov5_openvino_sync_dGPU.cpp: YOLOv5 Sync Infer C++ Demo on intel discreate GPU(A770)

### references
+ https://docs.ultralytics.com/quick-start/
+ https://github.com/ultralytics/yolov5/releases/tag/v6.1
+ https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html#
+ https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109
