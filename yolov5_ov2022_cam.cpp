#include <fstream>                   //C++ 文件操作
#include <iostream>                  //C++ input & output stream
#include <sstream>                   //C++ String stream, 读写内存中的string对象
#include <opencv2\opencv.hpp>        //OpenCV 头文件

#include <openvino\openvino.hpp>     //OpenVINO >=2022.1

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
//OpenVINO IR模型文件路径
string ir_filename = "D:/yolov5/yolov5s.onnx";

// @brief 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络
// @param input_tensor 输入节点的tensor
// @param inpt_image 输入图片数据
void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
	// 获取输入节点要求的输入图片数据的大小
	ov::Shape tensor_shape = input_tensor.get_shape();
	const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
	const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
	const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
	// 读取节点数据内存指针
	float* input_tensor_data = input_tensor.data<float>();
	// 将图片数据填充到网络中
	// 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
	for (size_t c = 0; c < channels; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
			}
		}
	}
}

int main(int argc, char** argv) {

	//创建OpenVINO Core
	Core core;
	CompiledModel compiled_model = core.compile_model(ir_filename, "AUTO");
	InferRequest infer_request = compiled_model.create_infer_request();

	// 从USB Webcam采集数据
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) {
		cout << "Exit!webcam fails to open!" << endl;
		return -1;
	}

	// 获取输入节点tensor信息
	Tensor input_image_tensor = infer_request.get_tensor("images");
	int input_h = input_image_tensor.get_shape()[2]; //获得"images"节点的Height
	int input_w = input_image_tensor.get_shape()[3]; //获得"images"节点的Width
	cout << "input_h:" << input_h << "; input_w:" << input_w << endl;
	cout << "input_image_tensor's element type:" << input_image_tensor.get_element_type() << endl;
	cout << "input_image_tensor's shape:" << input_image_tensor.get_shape() << endl;
	// 获取输出节点tensor信息
	Tensor output_tensor = infer_request.get_tensor("output");
	int out_rows = output_tensor.get_shape()[1]; //获得"output"节点的out_rows
	int out_cols = output_tensor.get_shape()[2]; //获得"output"节点的Width
	cout << "out_cols:" << out_cols << "; out_rows:" << out_rows << endl;

	//连续采集处理循环
	while (true) {

		Mat frame;
		cap >> frame;
		int64 start = cv::getTickCount();

		// 图像预处理
		int w = frame.cols;
		int h = frame.rows;
		int _max = std::max(h, w);
		cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
		cv::Rect roi(0, 0, w, h);
		frame.copyTo(image(roi));
		cvtColor(image, image, COLOR_BGR2RGB); //交换RB通道

		float x_factor = image.cols / input_w;
		float y_factor = image.rows / input_h;

		cv::Mat blob_image;
		resize(image, blob_image, cv::Size(input_w, input_h));
		blob_image.convertTo(blob_image, CV_32F);
		blob_image = blob_image / 255.0;

		// 将预处理后的图像数据填充到tensor数据内存中
		fill_tensor_data_image(input_image_tensor, blob_image);

		// 执行推理计算
		infer_request.infer();

		// 获得推理结果
		const ov::Tensor& output_tensor = infer_request.get_tensor("output");

		// 解析推理结果，YOLOv5 output format: cx,cy,w,h,score
		cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_tensor.data());
		
		std::vector<cv::Rect> boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;

		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < 0.4) {
				continue;
			}
			cv::Mat classes_scores = det_output.row(i).colRange(5, 85);
			cv::Point classIdPoint;
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
				cv::Rect box;
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
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
		for (size_t i = 0; i < indexes.size(); i++) {
			int index = indexes[i];
			int idx = classIds[index];
			cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
			cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
				cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}

		// 计算FPS
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << endl;
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", frame);

		char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}
	}

	cv::waitKey(0);
	cv::destroyAllWindows();
	
	return 0;
}
