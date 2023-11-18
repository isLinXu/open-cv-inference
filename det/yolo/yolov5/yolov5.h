//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_YOLOV5_H
#define OPEN_CV_INFERENCE_YOLOV5_H

#include <fstream>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class YoloV5Detector {
public:
    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    YoloV5Detector(bool is_cuda = false,
                   std::string classes_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/c++/classes.txt",
                   std::string model_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/c++/yolov5s.onnx");

    std::vector<std::string> load_class_list(std::string classes_path);

    void load_net(std::string model_path = "/Users/gatilin/CLionProjects/yolov5_onnx_dnn/c++/yolov5s.onnx");

    cv::Mat format_yolov5(const cv::Mat &source);

    void detect(cv::Mat &image, std::vector<Detection> &output, std::vector<std::string> &className);

    int infer(int argc, char **argv);

private:
    const std::vector<cv::Scalar> colors;
    const float INPUT_WIDTH;
    const float INPUT_HEIGHT;
    const float SCORE_THRESHOLD;
    const float NMS_THRESHOLD;
    const float CONFIDENCE_THRESHOLD;
    bool is_cuda;
    cv::dnn::Net net;
    std::vector<std::string> class_list;
};

#endif //OPEN_CV_INFERENCE_YOLOV5_H
