import cv2
import numpy as np
import time
import yaml
import torch
from openvino.runtime import Core, Tensor
# https://github.com/zhiqwang/yolov5-rt-stack
from yolort.v5 import non_max_suppression, scale_coords

# Load COCO Label from yolov5/data/coco.yaml
with open('./data/coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# Step1: Create OpenVINO Runtime Core
core = Core()
# Step2: Compile the Model for the dedicated device: CPU/GPU.0/GPU.1...
net = core.compile_model("yolov5s.xml", "GPU.1")

# get input node and output node
input_node = net.inputs[0]
output_node = net.outputs[0]

# Step 3. Create 1 Infer_request for current frame, 1 for next frame
infer_request_curr = net.create_infer_request()
infer_request_next = net.create_infer_request()

# color palette
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
#import the letterbox for preprocess the frame
from utils.augmentations import letterbox

# Get the current frame
frame_curr = cv2.imread("./data/images/bus.jpg")
# Preprocess the frame
letterbox_img_curr, _, _ = letterbox(frame_curr, auto=False)
# Normalization + Swap RB + Layout from HWC to NCHW
blob = Tensor(cv2.dnn.blobFromImage(letterbox_img_curr, 1/255.0, swapRB=True))  
# Transfer the blob into the model
infer_request_curr.set_tensor(input_node, blob)
# Start the current frame Async Inference
infer_request_curr.start_async()

while True:    
    # Calculate the end-to-end process throughput.
    start = time.time()
    # Get the next frame
    frame_next = cv2.imread("./data/images/zidane.jpg")
    # Preprocess the frame
    letterbox_img_next, _, _ = letterbox(frame_next, auto=False)
    # Normalization + Swap RB + Layout from HWC to NCHW
    blob = Tensor(cv2.dnn.blobFromImage(letterbox_img_next, 1/255.0, swapRB=True))  
    # Transfer the blob into the model
    infer_request_next.set_tensor(input_node, blob)
    # Start the next frame Async Inference
    infer_request_next.start_async()
    # wait for the current frame inference result
    infer_request_curr.wait()

    # Get the inference result from the output_node
    infer_result = infer_request_curr.get_tensor(output_node)
    # Postprocess the inference result
    data = torch.tensor(infer_result.data)
    # Postprocess of YOLOv5:NMS
    dets = non_max_suppression(data)[0].numpy()
    bboxes, scores, class_ids= dets[:,:4], dets[:,4], dets[:,5]
    # rescale the coordinates
    bboxes = scale_coords(letterbox_img_curr.shape[:-1], bboxes, frame_curr.shape[:-1]).astype(int)

    # show bbox of detections
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(frame_curr, (bbox[0],bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.rectangle(frame_curr, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), color, -1)
        cv2.putText(frame_curr, class_list[class_id], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    end = time.time() 

    # show FPS
    fps = (1 / (end - start)) 
    fps_label = "Throughput: %.2f FPS" % fps
    cv2.putText(frame_curr, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label+ "; Detections: " + str(len(class_ids)))
    cv2.imshow("Async API demo", frame_curr)

    # Swap the infer request 
    infer_request_curr, infer_request_next = infer_request_next, infer_request_curr
    frame_curr = frame_next
    letterbox_img_curr = letterbox_img_next
        
    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        cv2.destroyAllWindows()
        break