//
// Created by gatilin on 2023/11/18.
//
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utility>
#include "yolov5.h"

YoloV5Detector::YoloV5Detector(bool is_cuda,std::string classes_path,std::string model_path)
        : colors({cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)}),
          INPUT_WIDTH(640.0),
          INPUT_HEIGHT(640.0),
          SCORE_THRESHOLD(0.2),
          NMS_THRESHOLD(0.4),
          CONFIDENCE_THRESHOLD(0.4),
          is_cuda(is_cuda) {
    /**
     * @param is_cuda: use cuda or not
     * @param classes_path: path to classes.txt
     * @param model_path: path to model
     * @return: void
     * 1. Load class list
     * 2. Load model
     * 3. Set backend and target
     * 4. Set input size
     * 5. Set confidence threshold
     * 6. Set nms threshold
     * 7. Set score threshold
     * 8. Set input blob
     * 9. Set output blob
     */
    class_list = load_class_list(std::move(classes_path));
    load_net(std::move(model_path));
}

std::vector<std::string> YoloV5Detector::load_class_list(const std::string& classes_path) {
    /**
     * @param classes_path: path to classes.txt
     * @return: class list
     */
    std::ifstream ifs(classes_path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void YoloV5Detector::load_net(std::string model_path) {
    /**
     * @param model_path: path to model
     * @return: void
     * 1. Read model
     * 2. Set backend and target
     */
    auto result = cv::dnn::readNet(model_path);
    if (is_cuda) {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_HALIDE);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat YoloV5Detector::format_yolov5(const cv::Mat &source) {
    /**
     * @param source: input image
     * @return: formatted image
     */
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void YoloV5Detector::detect(cv::Mat &image, std::vector<Detection> &output, std::vector<std::string> &className) {
    /**
     * @param image: input image
     * @param output: output vector of detections
     * @param className: output vector of class names
     * @return: void
     * 1. Format image to square
     * 2. Create blob
     * 3. Set blob as input of network
     * 4. Forward
     * 5. Post process
     * 6. Draw
     * 7. Return
     */
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255.,
                           cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true,false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    auto *data = (float *) outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            // std::cout << "confidence:" << confidence << std::endl;
            float *classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                // std::cout << "class_id.x:" << class_id.x << std::endl;
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}


int YoloV5Detector::infer(int argc, char **argv){
    std::string image_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/input_image/2007_005331.jpg";
    std::string model_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/c++/yolov5s.onnx";
    std::string classes_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/c++/classes.txt";

    cv::Mat frame = cv::imread(image_path);

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(model_path);

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    std::vector<Detection> output;
    detect(frame, output,class_list);

    int detections = output.size();

    for (int i = 0; i < detections; ++i)
    {

        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);
        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("output", frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}