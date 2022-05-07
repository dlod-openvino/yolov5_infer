#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace ov;
using namespace cv;

//OpenVINO IR模型文件路径
string model_file = "D:/yolov5/yolov5s.onnx";
//测试图片路径
string image_file = "D:/yolov5/data/images/zidane.jpg";

int main(int argc, char** argv) {
	//1.创建OpenVINO Runtime Core对象
	Core core;
	//2.载入并编译模型
	CompiledModel compiled_model = core.compile_model(model_file, "AUTO");
	//输出输入模型的输入输出节点名字
	auto input_node_name = compiled_model.input(0).get_any_name();
	auto output_node_name = compiled_model.output(0).get_any_name();
	cout << "input node name: " << input_node_name << "; output node name: " << output_node_name << endl;
	//读取图片并显示
	Mat image = imread(image_file);
	imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", image);

	cv::waitKey(0);
	cv::destroyAllWindows();
    return 0;
}