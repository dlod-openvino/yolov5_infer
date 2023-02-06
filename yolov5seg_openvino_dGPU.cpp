#include <iostream>
#include <string>

#include <openvino/openvino.hpp> //include openvino runtime header files
#include <opencv2/opencv.hpp>    //opencv header file

/* ---------  Please modify the path of yolov5 model and image -----------*/
std::string model_file = "C:/Users/NUC/Desktop/yolov5/yolov5s-seg.xml";
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

float sigmoid_function(float a)
{
    float b = 1. / (1. + exp(-a));
    return b;
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
    cv::RNG rng;
    cv::Mat img = cv::imread(image_file); //Load a picture into memory
    cv::Mat masked_img;
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
    auto detect = infer_request.get_output_tensor(0);
    auto detect_shape = detect.get_shape();
    std::cout << "The shape of Detection tensor:"<< detect_shape << std::endl;
    auto proto = infer_request.get_output_tensor(1);
    auto proto_shape = proto.get_shape();
    std::cout << "The shape of Proto tensor:" << proto_shape << std::endl;

    // --------- Do the Post Process

    // Detect Matrix: 25200 x 117  
    cv::Mat detect_buffer(detect_shape[1], detect_shape[2], CV_32F, detect.data());
    // Proto Matrix:  1x32x160x160 => 32 x 25600
    cv::Mat proto_buffer(proto_shape[1], proto_shape[2] * proto_shape[3], CV_32F, proto.data());

    // -------- Step 8. Post-process the inference result -----------
    float conf_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<float> confidences;
    std::vector<cv::Mat> masks;
    // cx,cy,w,h,confidence,c1,c2,...c80
    float scale = paddings[0];
    for (int i = 0; i < detect_buffer.rows; i++) {
        float confidence = detect_buffer.at<float>(i, 4);
        if (confidence < conf_threshold) {
            continue;
        }
        cv::Mat classes_scores = detect_buffer.row(i).colRange(5, 85);
        cv::Point class_id;
        double score;
        cv::minMaxLoc(classes_scores, NULL, &score, NULL, &class_id);

        // class score: 0~1
        if (score > 0.25)
        {
            cv::Mat mask = detect_buffer.row(i).colRange(85, 117);
            float cx = detect_buffer.at<float>(i, 0);
            float cy = detect_buffer.at<float>(i, 1);
            float w = detect_buffer.at<float>(i, 2);
            float h = detect_buffer.at<float>(i, 3);
            int left = static_cast<int>((cx - 0.5 * w - paddings[2]) / scale);
            int top = static_cast<int>((cy - 0.5 * h - paddings[1]) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);
            cv::Rect box;
            box.x = left;
            box.y = top;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            class_ids.push_back(class_id.x);
            class_scores.push_back(score);
            confidences.push_back(confidence);
            masks.push_back(mask);
        }
    }
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    cv::Mat rgb_mask = cv::Mat::zeros(img.size(), img.type());

    // -------- Step 8. Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        int class_id = class_ids[index];
        cv::Rect box = boxes[index];
        int x1 = std::max(0, box.x);
        int y1 = std::max(0, box.y);
        int x2 = std::max(0, box.br().x);
        int y2 = std::max(0, box.br().y);

        cv::Mat m = masks[index] * proto_buffer;
        for (int col = 0; col < m.cols; col++) {
            m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
        }
        cv::Mat m1 = m.reshape(1, 160); // 1x25600 -> 160x160

        int mx1 = std::max(0, int((x1 * scale + paddings[2]) * 0.25));
        int mx2 = std::max(0, int((x2 * scale + paddings[2]) * 0.25));
        int my1 = std::max(0, int((y1 * scale + paddings[1]) * 0.25));
        int my2 = std::max(0, int((y2 * scale + paddings[1]) * 0.25));
        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;
        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
        for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 1.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }
        rm = rm * rng.uniform(0, 255);
        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= img.rows) {
            y2 = img.rows - 1;
        }
        if ((x1 + det_mask.cols) >= img.cols) {
            x2 = img.cols - 1;
        }

        cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
        add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);

        cv::rectangle(img, boxes[index], colors[class_id % 6], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]);
        cv::putText(img, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, colors[class_id % 6]);
        cv::addWeighted(img, 0.5, rgb_mask, 0.5, 0, masked_img);
    }

    cv::namedWindow("YOLOv5-Seg OpenVINO Inference C++ Demo");
    cv::imshow("YOLOv5-Seg OpenVINO Inference C++ Demo", masked_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}