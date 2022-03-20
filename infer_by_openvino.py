import cv2
import numpy as np
import time
import yaml
from openvino.inference_engine import IECore # the version of openvino <= 2021.4.2

# 载入COCO Label
with open('./coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 目标检测函数，返回检测结果
def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    result = net.infer({"images": blob})
    preds = result["output"]
    return preds

# YOLOv5的后处理函数，解析模型的输出
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    #print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

# 按照YOLOv5要求，先将图像长:宽 = 1:1，多余部分填充黑边
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


# 载入yolov5s onnx模型
model_path = "./yolov5s.onnx"
# Read yolov5s onnx model with OpenVINO API
ie = IECore()  #Initialize IECore
net = ie.load_network(network=model_path, device_name="AUTO")

# 开启Webcam，并设置为1280x720
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 调色板
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# 开启检测循环
while True:
    start = time.time()
    _, frame = cap.read()
    if frame is None:
        print("End of stream")
        break
    # 将图像按最大边1:1放缩
    inputImage = format_yolov5(frame)
    # 执行推理计算
    outs = detect(inputImage, net)
    # 拆解推理结果
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    # 显示检测框bbox
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
    
    # 显示推理速度FPS
    end = time.time()
    inf_end = end - start
    fps = 1 / inf_end
    fps_label = "FPS: %.2f" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label+ "; Detections: " + str(len(class_ids)))
    cv2.imshow("output", frame)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break
