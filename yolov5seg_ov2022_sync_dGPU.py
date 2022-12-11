import numpy as np
import cv2, yaml, torch
from openvino.runtime import Core
from utils.general import non_max_suppression
from utils.augmentations import letterbox  

# Load COCO Label from yolov5/data/coco.yaml
with open('./data/coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']
# color palette
colors = [(200, 150, 0), (0, 200, 0), (0, 200, 150), (200, 0, 0)]

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
# Rescale coords (xyxy) from according to r and (dh, dw) from letterbox
def rescale_coords(ratio, pad, coords):
    # Rescale coords (xyxy) from according to r and (dh, dw) from letterbox
    coords[:, [1, 3]] -= pad[0]  # H padding
    coords[:, [0, 2]] -= pad[1]  # W padding
    coords[:, :4] /= ratio
    return coords

# Step1: Create OpenVINO Runtime Core
core = Core()
# Step2: Compile the Model, using dGPU A770m
net = core.compile_model("yolov5s-seg.xml", "GPU.1")
output0, output1 = net.outputs[0],net.outputs[1]
b,n,input_h,input_w = net.inputs[0].shape

# Step3: Preprocess the image before inference
frame =cv2.imread("./data/images/zidane.jpg")
fh, fw, fc = frame.shape
im, r, (dw, dh)= letterbox(frame, new_shape=(input_h,input_w), auto=False) # Resize to new shape by letterbox
blob = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
blob = np.ascontiguousarray(blob)  # contiguous
blob = np.float32(blob) / 255.0    # 0 - 255 to 0.0 - 1.0
blob = blob[None]  # expand for batch dim

# Step4: Do the inference
outputs = net([blob])
pred, proto = outputs[output0], outputs[output1]

# Step5: Postprocess the inference result and visulize it.
pred = torch.tensor(pred)
pred = non_max_suppression(pred, nm=32)[0].numpy() #(n,38) tensor per image [xyxy, conf, cls, masks]
# (n,38) tensor per image [xyxy, conf, cls, masks]
bboxes, confs, class_ids, masks= pred[:,:4], pred[:,4], pred[:,5], pred[:,6:]

# Extract the mask of the detected object
proto = np.squeeze(proto)
proto = np.reshape(proto, (32,-1))
obj_masks = np.matmul(masks,proto)
obj_masks = np.reshape(sigmoid(obj_masks), (-1, 160, 160))

masks_roi = []
for obj_mask, bbox in zip(obj_masks, bboxes):
    mx1 = max(0, np.int32((bbox[0] * 0.25)))
    my1 = max(0, np.int32((bbox[1] * 0.25)))
    mx2 = max(0, np.int32((bbox[2] * 0.25)))
    my2 = max(0, np.int32((bbox[3] * 0.25)))
    masks_roi.append(obj_mask[my1:my2,mx1:mx2])

# rescale the coordinates
bboxes = rescale_coords(r[0], (dh, dw), bboxes).astype(int)

color_mask = np.zeros((fh, fw, 3), dtype=np.uint8)
black_mask = np.zeros((fh, fw), dtype=np.float32)
mv = cv2.split(color_mask)
#Show bboxes and object's masks
for bbox, conf, class_id, mask_roi in zip(bboxes, confs, class_ids, masks_roi):
    x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]    
    # Draw Mask
    color = colors[int(class_id) % len(colors)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)
    # Draw mask of the detected objects
    result_mask = cv2.resize(mask_roi, (bbox[2]-bbox[0], bbox[3]-bbox[1]))
    result_mask[result_mask > 0.5] = 1.0
    result_mask[result_mask <= 0.5] = 0.0
    rh, rw = result_mask.shape
    rh, rw = result_mask.shape
    if (y1+rh) >= fh:
        rh = fh - y1
    if (x1+rw) >= fw:
        rw = fw - x1
    black_mask[y1:y1+rh, x1:x1+rw] = result_mask[0:rh, 0:rw]
    mv[2][black_mask == 1], mv[1][black_mask == 1], mv[0][black_mask == 1] = \
            [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
    # Draw Label
    cv2.putText(frame, class_list[class_id]+":"+ "%0.2f"%conf, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

# Add Masks to the frame
color_mask = cv2.merge(mv)
dst = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
# Show the frame with the masks
cv2.imshow("YOLOv5-Seg inference on Intel dGPU Demo", dst)
cv2.waitKey()
cv2.destroyAllWindows()