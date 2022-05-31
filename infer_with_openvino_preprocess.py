# YOLOv5s.onnx OpenVINO2022.1 Inference Demo with OpenVINO preprocess
import time
import numpy as np

# Please modify the model path and image path
model_path = "yolov5s.onnx" 
image_path = "data/images/zidane.jpg"
niter = 500                             # Number of iterations
# Load COCO Label
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

predictions = np.ndarray([1, 25200, 85]) # output shape of yolov5s

# Step1: Instance PrePostProcessor object
from openvino.runtime import Core, InferRequest, AsyncInferQueue, Layout, Type
from openvino.preprocess import PrePostProcessor, ColorFormat
core = Core()
model = core.read_model(model_path)
ppp = PrePostProcessor(model)

# Step2: Declare input data information:
ppp.input().tensor() \
    .set_color_format(ColorFormat.BGR) \
    .set_element_type(Type.u8) \
    .set_layout(Layout('NHWC'))  

# Step3: Specify actual model layout
ppp.input().model().set_layout(Layout('NCHW'))

# Step4: Set output tensor information:
# - precision of tensor is supposed to be 'f32'
ppp.output().tensor().set_element_type(Type.f32)

# Step5: Apply preprocessing modifing the original 'model'
# - Precision from u8 to f32
# - color plane from BGR to RGB
# - subtract mean
# - divide by scale factor
# - Layout conversion will be done automatically as last step
ppp.input().preprocess() \
    .convert_element_type(Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .mean([0.0, 0.0, 0.0]) \
    .scale([255.0, 255.0, 255.0]) 

# Step6: Integrate preprocessing steps into model
print(f'Build preprocessor: {ppp}')
model = ppp.build()
compiled_model = core.compile_model(model, "GPU")

# Prepare the input data
import cv2
from utils.augmentations import letterbox
img = cv2.imread(image_path)
letterbox_img, ratio, (dw, dh) = letterbox(img, auto=False)
# Change shape from HWC to NHWC
input_tensor = np.expand_dims(letterbox_img, axis=0)

start_time = time.time()
# Do the inference in the async mode
opt_nireq = compiled_model.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
print('OPTIMAL_NUMBER_OF_INFER_REQUESTS:', opt_nireq)
ireq_queue = AsyncInferQueue(compiled_model, opt_nireq)

def completion_callback(infer_request: InferRequest, image_path: str) -> None:
    global predictions
    predictions=next(iter(infer_request.results.values()))

ireq_queue.set_callback(completion_callback)

for n in range(niter):
    # the preprocess by OpenCV
    # blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, swapRB=True)
    ireq_queue.start_async([input_tensor])  
    
ireq_queue.wait_all()
end_time = time.time()
print(f"The total infer time of {niter} times: {end_time - start_time}s.")

# Process inference results
import numpy as np
class_ids = []
confidences = []
boxes = []

for pred in predictions:
    for i, det in enumerate(pred):
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

filtered_ids = []
filered_confidences = []
filtered_boxes = []

for i in indexes:
    filtered_ids.append(class_ids[i])
    filered_confidences.append(confidences[i])
    filtered_boxes.append(boxes[i])

# colorbox
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
# Show bbox
for (class_id, confidence, box) in zip(filtered_ids, filered_confidences, filtered_boxes):
    color = colors[int(class_id) % len(colors)]
    cv2.rectangle(img, box, color, 2)
    cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
    cv2.putText(img, class_names[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    print(class_names[class_id],box)
    
print("Detections: " + str(len(filtered_ids)))
cv2.imshow("YOLOv5+OpenVINO_2022.1 Preprocess API Demo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
