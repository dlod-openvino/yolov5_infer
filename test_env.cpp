#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace ov;
using namespace cv;

//OpenVINO IRģ���ļ�·��
string model_file = "D:/yolov5/yolov5s.onnx";
//����ͼƬ·��
string image_file = "D:/yolov5/data/images/zidane.jpg";

int main(int argc, char** argv) {
	//1.����OpenVINO Runtime Core����
	Core core;
	//2.���벢����ģ��
	CompiledModel compiled_model = core.compile_model(model_file, "AUTO");
	//�������ģ�͵���������ڵ�����
	auto input_node_name = compiled_model.input(0).get_any_name();
	auto output_node_name = compiled_model.output(0).get_any_name();
	cout << "input node name: " << input_node_name << "; output node name: " << output_node_name << endl;
	//��ȡͼƬ����ʾ
	Mat image = imread(image_file);
	imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", image);

	cv::waitKey(0);
	cv::destroyAllWindows();
    return 0;
}