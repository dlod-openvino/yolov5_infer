# Integrate preprocess funciton into YOLOv5s model and save it as YOLOv5s.xml
from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat

# Please modify the model path
model_path = "yolov5s.onnx" 

# Step1: Create OpenVINO Runtime Core
core = Core()

# Step2: Read the Model and Load the model to a device with specific configuration properties
model = core.read_model(model_path)

# Step3: integrate preprocess function into the model by OpenVINO PrePostProcessor
# Normalization + Swap RB + Layout from HWC to NCHW
# ======== Preprocessing by OpenVINO ================
ppp = PrePostProcessor(model)

# 1) Declare input data information:
# - input() provides information about a single model input
# - precision of tensor is supposed to be 'f32'
# - layout of data is 'NHWC'
ppp.input().tensor() \
    .set_color_format(ColorFormat.BGR) \
    .set_element_type(Type.u8) \
    .set_layout(Layout('NHWC'))  

# 2) Specify actual model layout
ppp.input().model().set_layout(Layout('NCHW'))

# 3) Set output tensor information:
# - precision of tensor is supposed to be 'f32'
ppp.output().tensor().set_element_type(Type.f32)

# 4) Apply preprocessing modifing the original 'model'
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

# Dump preprocessor information
print(f'Build preprocessor: {ppp}')
model = ppp.build()

# Step4: Save the Model with preprocess
from openvino.offline_transformations import serialize
serialize(model, 'yolov5s.xml', 'yolov5s.bin')
