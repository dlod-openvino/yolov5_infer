import cv2
import time
import yaml
import torch
from openvino.runtime import Core

# https://github.com/zhiqwang/yolov5-rt-stack
from yolort.v5 import non_max_suppression, scale_coords

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
    start = time.time() # total excution time =  preprocess + infer + postprocess
    frame = cv2.imread("./data/images/zidane.jpg")
    # preprocess frame by letterbox
    letterbox_img, _, _= letterbox(frame, auto=False)
    # Normalization + Swap RB + Layout from HWC to NCHW
    blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, swapRB=True)
    # Step 3: Do the inference
    outs = torch.tensor(net([blob])[output_node]) 
    # Postprocess of YOLOv5:NMS
    dets = non_max_suppression(outs)[0].numpy()
    bboxes, scores, class_ids= dets[:,:4], dets[:,4], dets[:,5]
    # rescale the coordinates
    bboxes = scale_coords(letterbox_img.shape[:-1], bboxes, frame.shape[:-1]).astype(int)

    #Show bbox
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[2], bbox[1]), color, -1)
        cv2.putText(frame, class_list[class_id], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    end = time.time()

    # show FPS
    fps = (1 / (end - start)) 
    fps_label = "Throughput: %.2f FPS" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(fps_label+ "; Detections: " + str(len(class_ids)))
    cv2.imshow("Sync API demo", frame)

    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        cv2.destroyAllWindows()
        break

