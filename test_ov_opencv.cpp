#include <iostream>
#include <string>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

int main(int argc, char *argv[])
{
    // -------- Get OpenVINO runtime version --------
    std::cout << ov::get_openvino_version().description << ':' << ov::get_openvino_version().buildNumber << std::endl;

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Get list of available devices --------
    std::vector<std::string> availableDevices = core.get_available_devices();

    // -------- Step 3. Query and print supported metrics and config keys --------
    std::cout << "Available devices: " << std::endl;
    for (auto &&device : availableDevices)
    {
        std::cout << device << std::endl;
    }

    // -------- Step 4. Read a picture file and show by OpenCV --------
    cv::Mat img = cv::imread("zidane.jpg"); // Load a picture into memory
    cv::namedWindow("Test OpenVINO & OpenCV IDE", 0);
    cv::imshow("Test OpenVINO & OpenCV IDE", img);
    std::cout << "Image width: " << img.cols << " height: " << img.rows << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
