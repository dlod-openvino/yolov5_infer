import cv2, torch, os, statistics
from time import perf_counter
import numpy as np
import logging as log

from openvino.runtime import Core, get_version, AsyncInferQueue, InferRequest
from openvino.runtime.utils.types import get_dtype
from utils.augmentations import letterbox  

# https://github.com/zhiqwang/yolov5-rt-stack
from yolort.v5 import non_max_suppression, scale_coords


def preprocess(frame):
    # Preprocess the frame
    letterbox_im, _, _= letterbox(frame, auto=False) # preprocess frame by letterbox
    im = letterbox_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.float32(im) / 255.0    # 0 - 255 to 0.0 - 1.0
    blob = im[None]  # expand for batch dim
    return blob, letterbox_im.shape[:-1], frame.shape[:-1]


def postprocess(ireq: InferRequest, user_data: tuple):
    result = ireq.results[ireq.model_outputs[0]]
    dets = non_max_suppression(torch.tensor(result))[0].numpy()
    bboxes, scores, class_ids= dets[:,:4], dets[:,4], dets[:,5]
    # rescale the coordinates
    bboxes = scale_coords(user_data[1], bboxes, user_data[2]).astype(int)
    print(user_data[0],"\t"+f"{ireq.latency:.3f}"+"\t", class_ids)
    return 

# Step1： Initialize OpenVINO Runtime Core
core = Core()

# Step2:  Build compiled model
device = device = ['GPU.0', 'GPU.1', 'CPU', 'AUTO', 'AUTO:GPU,-CPU'][0]
cfgs = {}
cfgs['PERFORMANCE_HINT'] = ['THROUGHPUT', 'LATENCY', 'CUMULATIVE_THROUGHPUT'][0]
net = core.compile_model("yolov5s.xml",device,cfgs)
output_node = net.outputs[0]
b,n,input_h,input_w = net.inputs[0].shape

# Step3:  Initialize InferQueue
ireqs = AsyncInferQueue(net)
print('Number of infer requests in InferQueue:', len(ireqs))
# Step3.1: Set unified callback on all InferRequests from queue's pool
ireqs.set_callback(postprocess)

# Step4:  Read the images
image_folder = "./data/images/"
image_files= os.listdir(image_folder)
print(image_files)
frames = []
for image_file in image_files:
    frame = cv2.imread(os.path.join(image_folder, image_file))
    frames.append(frame)

# 4.1 Warm up
for id, _ in enumerate(ireqs):
    # Preprocess the frame
    start = perf_counter()
    blob, letterbox_shape, frame_shape = preprocess(frames[id % 4])
    end = perf_counter()
    print(f"Preprocess {id}: {(end-start):.4f}.")
    # Run asynchronous inference using the next available InferRequest from the pool
    ireqs.start_async({0:blob},(id, letterbox_shape, frame_shape))
ireqs.wait_all()

# Step5:  Benchmark the Async Infer
start = perf_counter()
in_fly = set()
latencies = []
niter = 16
for i in range(niter):
    # Preprocess the frame
    blob, letterbox_shape, frame_shape = preprocess(frames[i % 4]) 
    idle_id = ireqs.get_idle_request_id()
    if idle_id in in_fly:
        latencies.append(ireqs[idle_id].latency)
    else:
        in_fly.add(idle_id)
    # Run asynchronous inference using the next available InferRequest from the pool 
    ireqs.start_async({0:blob},(i, letterbox_shape, frame_shape) )
ireqs.wait_all()
duration = perf_counter() - start

# Step6:  Report results
fps = niter / duration
log.info(f'Count:          {niter} iterations')
log.info(f'Duration:       {duration * 1e3:.2f} ms')
log.info('Latency:')
log.info(f'    Median:     {statistics.median(latencies):.2f} ms')
log.info(f'    Average:    {sum(latencies) / len(latencies):.2f} ms')
log.info(f'    Min:        {min(latencies):.2f} ms')
log.info(f'    Max:        {max(latencies):.2f} ms')
log.info(f'Throughput: {fps:.2f} FPS')