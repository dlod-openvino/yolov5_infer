#version: python 3.x 测试过
#用于模拟一个基于mqtt协议传输OpenVINO推理结果

import paho.mqtt.client as mqtt
import json
import time
import queue as Queue
import threading,random
import cv2
import numpy as np
import time
import yaml
# from openvino.inference_engine import IECore # the version of openvino <= 2021.4.2
from openvino.runtime import Core # the version of openvino >= 2022.1

BROKER_HOST_ADDR   = "localhost"
BROKER_HOST_PORT   = 1883
USERNAME    = "huaqiaoz" #非安全模式忽略用户名秘密
PWD         = "1234" #非安全模式忽略用户名秘密
#cmd topic本质上就是你的设备监听的topic，
#也是在UI上添加device的时候，地址中所填数据，和用户名密码等一起组成当前设备的唯一标识。
CMD_TOPIC   = "CommandTopic"
RESPONSE_TOPIC = "ResponseTopic"
DATA_TOPIC  = "DataTopic"

# 载入COCO Label
with open('./coco.yaml','r', encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
class_list = result['names']

# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# 载入yolov5s onnx模型
model_path = "./yolov5s.onnx"
# Read yolov5s onnx model with OpenVINO API
ie = Core()  #Initialize IECore
net = ie.compile_model(model=model_path, device_name="AUTO")

# 开启Webcam，并设置为1280x720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 调色板
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# 基于OpenVINO的YOLOv5s.onnx推理计算子函数
# 目标检测函数，返回检测结果
def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    #result = net.infer({"images": blob})
    #preds = result["output"]
    preds = net([blob])[next(iter(net.outputs))] # API version>=2022.1
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

globalQueue = Queue.Queue()

def gen():
       return round(random.uniform(0, 50),2)

def send_data():
    #java版本, name的值为添加的设备名
    #data = {"randnum":520.1314,"name":"mqtt-device-01"}

    #go版本, name的值为添加的设备名（required）, go版本的区别是必须带上cmd字段
    # data = {"randnum":520.1314,"name":"mqtt-device-01","cmd":"randnum","method":"get"}
    
    #v2.1.0版本数据格式
    #数据格式 data = {"randnum":520.1314,"name":"mqtt-device-01","cmd":"randnum","method":"get"}
    data = {"randnum":gen(),"name":"mqtt-device-01","cmd":"randnum", "method":"get"}
    print("sending data actively! " + json.dumps(data))
    client.publish(DATA_TOPIC,json.dumps(data) , qos=0, retain=False)

class SendDataActiveServer(threading.Thread):
    def __init__(self,threadID,name,queue):
        super(SendDataActiveServer,self).__init__()
        self.threadID = threadID
        self.name = name
        self.queue = queue
        self.active = False

    def run(self):
        while 1==1 :
          #v2.1.0 接收到的bool值自动自动转换为python能识别的bool值，无须二次转换
          if self.active:
             send_data()
             time.sleep(1)
             self.getItemFromQueue()
          else:
             time.sleep(1)
             self.getItemFromQueue()

    def getItemFromQueue(self):
        try:
          if self.queue.get(block=False):
             self.active = True
          else:
             self.active = False
        except Queue.Empty:
          #quene.get()方法在队列中为空是返回异常，捕获异常什么都不做，保持active原状
          time.sleep(0.1)

#当接收到命令，响应命令
#v2.1.0 get命令格式： {"cmd":"ping","method":"get","uuid":"f46ae5c7-2a08-4f56-a38b-d9e122d255d0"}
#v2.1.0 set命令格式 {"cmd":"ping","ping":"ping_set_value","method":"set","uuid":"95d559ad-7140-4d62-8dd8-6465c37fec10"}
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload)+'\n')
    d = json.loads(msg.payload)

    if d['cmd'] == "message":
       print("This is message cmd")
       if d['method'] == "get": # 执行OpenVINO推理程序，并拿到结果
         _, frame = cap.read()
         if frame is None:
            print("End of stream")
         # 将图像按最大边1:1放缩
         inputImage = format_yolov5(frame)
         # 执行推理计算
         outs = detect(inputImage, net)
         # 拆解推理结果
         class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
         msg = ""
         for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            msg += f"detected: {class_list[classid]}:{confidence:.2f}, at xmin:{box[0]}, ymin:{box[1]}, xmax:{box[2]}, ymax:{box[3]}.\n"
         d['message'] = msg
         print(msg)
       elif d['method'] == "set":
          d['result'] = "set successed."

    if d['cmd'] == "ping":
       print("This is ping cmd")
       d['ping'] = "pong"

    if d['cmd'] == "randnum":
       print("This is randnum cmd")
       d['randnum'] = gen()

    if d['cmd'] == "collect" and d['method'] == "set":
       print("This is collect set cmd")
       d['result'] = "set successed."
       #param的值是true或false,且是字符串类型, 
       #globalQueue.put(d['param'])
       #v2.1.0命令格式中移除了param属性，使用device resource name为实际参数名
       globalQueue.put(d['collect'])
    elif d['cmd'] == "collect" and d['method'] == "get":
       print("This is collect get cmd")
       d['collect'] = thread.active

    print(json.dumps(d))
    client.publish(RESPONSE_TOPIC, json.dumps(d))

def on_connect(client, userdata, flags, rc):
    print("Connected success with result code "+str(rc))
    #监听命令
    client.subscribe(CMD_TOPIC)



client = mqtt.Client()
client.username_pw_set(USERNAME, PWD)
client.on_message = on_message
client.on_connect = on_connect

client.connect(BROKER_HOST_ADDR, BROKER_HOST_PORT, 60)

#开始独立线程用于主动发送数据
thread = SendDataActiveServer("Thread-1", "SendDataServerThread", globalQueue)
thread.setDaemon(True)
thread.start()

client.loop_forever()
