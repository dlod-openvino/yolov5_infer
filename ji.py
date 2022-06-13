# 当发起测试时，系统会先调用ji.py中的init(),并将测试图片逐次送入process_image接口

import json
import sys
import numpy as np
import cv2
from openvino.runtime import Core
from utils.augmentations import letterbox

# 模型地址一定要和测试阶段选择的模型地址一致！！！
model_path='/project/train/models/train/exp4/weights/best.xml'   

core = Core()

# 初始化模型
def init():
    model = core.compile_model(model_path,"AUTO")
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):
    
    imgsz = [640,640]

    import yaml
    with open("data/coco.yaml","r",encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    class_names = result["names"]

    stride = 32
    fake_result = {}
    fake_result["model_data"] = {"objects": []}
    
    # letterbox preprocess
    img, ratio, (dw, dh)= letterbox(input_image, imgsz, stride,auto=False)
    # Normalization + Swap RB + Layout from HWC to NCHW
    blob = cv2.dnn.blobFromImage(img, 1/255.0, swapRB=True)

    # do the inference
    input_node = handle.inputs[0]
    output_node = handle.outputs[0]
    outs = handle({input_node:blob})[output_node]
    
    # Postprocess the results
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    for i in indexes:
        fake_result['model_data']['objects'].append({
                "xmin": int(boxes[i][0]),
                "ymin": int(boxes[i][1]),
                "xmax": int(boxes[i][0] + boxes[i][2]),
                "ymax": int(boxes[i][1] + boxes[i][3]),
                "confidence": confidences[i].tolist(),
                "name": class_names[class_ids[i]]
                })

    return json.dumps(fake_result, indent=4)

if __name__ == '__main__':
    # Test API
    img =cv2.imread('data/images/zidane.jpg')
    predictor = init()
    import time
    s = time.time()
    fake_result = process_image(predictor, img)
    e = time.time()
    print(fake_result)
    print(f"infer time:{(e-s)}s")
