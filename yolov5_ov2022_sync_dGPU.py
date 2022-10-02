import cv2
import numpy as np
import time
import yaml
from openvino.runtime import Core

# Load COCO Label from yolov5/data/coco.yaml
with open('./data/coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# Step1: Create OpenVINO Runtime Core
core = Core()
# Step2: Compile the Model, using dGPU
net = core.compile_model("yolov5s.xml", "GPU.1")
output_node = net.outputs[0]

# color palette
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
#import the letterbox for preprocess the frame
from utils.augmentations import letterbox

while True:    
    frame = cv2.imread("./data/images/zidane.jpg")
    start = time.time()
    # preprocess frame by letterbox
    letterbox_img, ratio, (dw, dh) = letterbox(frame, auto=False)
    # Normalization + Swap RB + Layout from HWC to NCHW
    blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, swapRB=True) 
    # Step 3: Do the inference
    outs = net([blob])[output_node] 
    end = time.time()

    # post-process the inference results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for i, det in enumerate(out):
            confidence = det[4]
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                x,y,w,h = det[0].item(),det[1].item(),det[2].item(),det[3].item()
                left = int((x - 0.5*w -dw) / ratio[0])
                top = int((y - 0.5*h - dh) / ratio[1])
                width = int(w / ratio[0])
                height = int(h / ratio[1])
                box = np.array([left, top, width, height])
                boxes.append(box)
    # NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    filtered_ids, filered_confidences, filtered_boxes = [],[],[]

    for i in indexes:
        filtered_ids.append(class_ids[i])
        filered_confidences.append(confidences[i])
        filtered_boxes.append(boxes[i])

    # show bbox
    for (classid, confidence, box) in zip(filtered_ids, filered_confidences, filtered_boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
    
    # show FPS
    inf_end = end - start
    fps = (1 / inf_end) 
    fps_label = "FPS: %.2f" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label+ "; Detections: " + str(len(class_ids)))
    cv2.imshow("output", frame)
    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        break