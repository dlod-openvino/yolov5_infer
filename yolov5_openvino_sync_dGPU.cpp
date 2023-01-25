#include <iostream>
#include <string>

#include <openvino/openvino.hpp> //include openvino runtime header files
#include <opencv2/opencv.hpp>    //opencv header file

/* ---------  Please modify the path of yolov5 model and image -----------*/
std::string model_file = "C:/Users/NUC/Desktop/yolov5/yolov5s.xml";
std::string image_file = "C:/Users/NUC/Desktop/yolov5/data/images/zidane.jpg";
std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                                   cv::Scalar(255, 255, 0) , cv::Scalar(0, 255, 255) , cv::Scalar(255, 0, 255) };
const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush" };

cv::Mat letterbox(cv::Mat& img, std::vector<float>& paddings, std::vector<int> new_shape = {640, 640})
{
    // Get current image shape [height, width]
    // Refer to https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L111

    int img_h = img.rows; 
    int img_w = img.cols;

    // Compute scale ratio(new / old) and target resized shape
    float scale = std::min(new_shape[1] * 1.0 / img_h, new_shape[0] * 1.0 / img_w);
    int resize_h = int(round(img_h * scale));
    int resize_w = int(round(img_w * scale));
    paddings[0] = scale;

    // Compute padding
    int pad_h = new_shape[1] - resize_h;
    int pad_w = new_shape[0] - resize_w;

    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    // divide padding into 2 sides
    float half_h = pad_h * 1.0 / 2;
    float half_w = pad_w * 1.0 / 2;
    paddings[1] = half_h;
    paddings[2] = half_w;

    // Compute padding boarder
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));

    // Add border
    cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resized_img;
}

int main(int argc, char* argv[]) {
    // -------- Get OpenVINO runtime version --------
    std::cout << ov::get_openvino_version().description << ':' << ov::get_openvino_version().buildNumber << std::endl;

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    auto compiled_model = core.compile_model(model_file, "GPU.1"); //GPU.1 is dGPU A770

    // -------- Step 3. Create an Inference Request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Step 4. Read a picture file and do the preprocess --------
    cv::Mat img = cv::imread(image_file); //Load a picture into memory
    std::vector<float> paddings(3);       //scale, half_h, half_w
    cv::Mat resized_img = letterbox(img, paddings); //resize to (640,640) by letterbox
    // BGR->RGB, u8(0-255)->f32(0.0-1.0), HWC->NCHW
    cv::Mat blob = cv::dnn::blobFromImage(resized_img, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true);

    // -------- Step 5. Feed the blob into the input node of YOLOv5 -------
    // Get input port for model with one input
    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:"<<output_shape << std::endl;
    // 25200 x 85 Matrix 
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output.data());

    // -------- Step 8. Post-process the inference result -----------
    float conf_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<float> confidences;
    // cx,cy,w,h,confidence,c1,c2,...c80
    for (int i = 0; i < output_buffer.rows; i++) {
        float confidence = output_buffer.at<float>(i, 4);
        if (confidence < conf_threshold) {
            continue;
        }
        cv::Mat classes_scores = output_buffer.row(i).colRange(5, 85);
        cv::Point class_id;
        double score;
        cv::minMaxLoc(classes_scores, NULL, &score, NULL, &class_id);

        // class score: 0~1
        if (score > 0.25)
        {
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            int left = static_cast<int>((cx - 0.5 * w - paddings[2]) / paddings[0]);
            int top = static_cast<int>((cy - 0.5 * h - paddings[1]) / paddings[0]);
            int width = static_cast<int>(w / paddings[0]);
            int height = static_cast<int>(h / paddings[0]);
            cv::Rect box;
            box.x = left;
            box.y = top;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            class_ids.push_back(class_id.x);
            class_scores.push_back(score);
            confidences.push_back(confidence);
        }
    }
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    // -------- Step 8. Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        int class_id = class_ids[index];
        cv::rectangle(img, boxes[index], colors[class_id % 6], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]);
        cv::putText(img, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, colors[class_id % 6]);
    }

    cv::namedWindow("YOLOv5 OpenVINO Inference C++ Demo", cv::WINDOW_AUTOSIZE);
    cv::imshow("YOLOv5 OpenVINO Inference C++ Demo", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}