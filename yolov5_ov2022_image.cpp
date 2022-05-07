#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace ov;
using namespace cv;

// COCO数据集的标签
vector<string> class_names = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant",
"stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe","backpack", "umbrella",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove","skateboard", "surfboard",
"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot",
"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse","remote",
"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

//模型文件路径
string model_file = "D:/yolov5/yolov5s.onnx";
//测试图片路径
string image_file = "D:/yolov5/data/images/zidane.jpg";

int main(int argc, char** argv) {
	//1.创建OpenVINO Runtime Core对象
	Core core;

	//2.载入并编译模型
	CompiledModel compiled_model = core.compile_model(model_file, "AUTO");

	//3.创建推理请求
	InferRequest infer_request = compiled_model.create_infer_request();

	//4.设置模型输入
	//4.1 获取模型输入节点形状
	Tensor input_node = infer_request.get_input_tensor();
	Shape tensor_shape = input_node.get_shape();

	//4.2读取图片并按照模型输入要求进行预处理
	Mat frame = imread(image_file, IMREAD_COLOR);
	//Lettterbox resize is the default resize method in YOLOv5.
	int w = frame.cols;
	int h = frame.rows;
	int _max = max(h, w);
	Mat image = Mat::zeros(Size(_max, _max), CV_8UC3);
	Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));
	//交换RB通道
	cvtColor(image, image, COLOR_BGR2RGB); 
	//计算缩放因子
	size_t num_channels = tensor_shape[1];
	size_t height = tensor_shape[2];
	size_t width = tensor_shape[3];
	float x_factor = image.cols / width;
	float y_factor = image.rows / height;

	int64 start = cv::getTickCount();
	//缩放图片并归一化
	Mat blob_image;
	resize(image, blob_image, cv::Size(width, height));
	blob_image.convertTo(blob_image, CV_32F);
	blob_image = blob_image / 255.0;

	// 4.3 将图像数据填入input tensor
	Tensor input_tensor = infer_request.get_input_tensor();
    // 获取指向模型输入节点数据块的指针
	float* input_tensor_data = input_tensor.data<float>();
	// 将图片数据填充到模型输入节点中
	// 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
	for (size_t c = 0; c < num_channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = blob_image.at<Vec<float, 3>>(h, w)[c];
			}
		}
	}
	
	// 5.执行推理计算
	infer_request.infer();

	// 6.处理推理计算结果
	// 6.1 获得推理结果
	const ov::Tensor& output = infer_request.get_tensor("output");
	const float* output_buffer = output.data<const float>();

	// 6.2 解析推理结果，YOLOv5 output format: cx,cy,w,h,score
	int out_rows = output.get_shape()[1]; //获得"output"节点的rows
	int out_cols = output.get_shape()[2]; //获得"output"节点的cols
	Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);

	vector<cv::Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;

	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < 0.5) {
			continue;
		}
		Mat classes_scores = det_output.row(i).colRange(5, 85);
		Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// 置信度 0～1之间
		if (score > 0.5)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}
	// NMS
	vector<int> indexes;
	dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		rectangle(frame, boxes[index], Scalar(0, 0, 255), 2, 8);
		rectangle(frame, Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			Point(boxes[index].br().x, boxes[index].tl().y), Scalar(0, 255, 255), -1);
		putText(frame, class_names[idx], Point(boxes[index].tl().x, boxes[index].tl().y - 10), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 0, 0));
	}

	// 计算FPS
	float t = (getTickCount() - start) / static_cast<float>(getTickFrequency());
	cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << endl;
	putText(frame, format("FPS: %.2f", 1.0 / t), Point(20, 40), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, 8);
	imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", frame);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
//https://towardsdatascience.com/yolo-v4-or-yolo-v5-or-pp-yolo-dad8e40f7109