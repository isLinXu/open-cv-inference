#include <fstream>

#include <opencv2/opencv.hpp>
#include <utility>
#include "det/yolo/yolov5/yolov5.h"
#include "utils/type_helper.h"
using namespace cv;

// 颜色设置
const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0),
                                         cv::Scalar(0, 255, 0),
                                         cv::Scalar(0, 255, 255),
                                         cv::Scalar(255, 0, 0) };

int yolov5_infer(bool is_cuda, const std::string& image_path, const std::string& classes_path, std::string weights_path, const std::string& output_path, bool is_show){
    /**
     * @param is_cuda: 是否使用GPU
     * @param image_path: 输入图片路径
     * @param classes_path: 类别文件路径
     * @param weights_path: 权重文件路径
     * @param output_path: 输出图片路径
     * @param is_show: 是否显示图片
     */
    cv::Mat frame = cv::imread(image_path);
    YoloV5Detector detector(is_cuda, classes_path,std::move(weights_path));
    auto classNames = detector.class_list;
    printf("class size: %zu\n", classNames.size());

    std::vector<YoloV5Detector::Detection> output;
    detector.detect(frame, output, classNames);

    int detections = output.size();

    for (int i = 0; i < detections; ++i)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        std::cout << "classId:" << classId << " label:" << classNames[classId].c_str() << " confidence:" << confidence << std::endl;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);
        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        std::string label_name = classNames[classId] + ":"+ floatToString(confidence);
        cv::putText(frame, label_name, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    if(is_show!=false){
        cv::imshow("output", frame);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    cv::imwrite(output_path, frame);
    return 0;
}

int main(int argc, char **argv) {
    std::cout << "Runing Open-CV-Inference..." << std::endl;
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    bool is_show = true;
    std::string image_path = "../images/zidane.jpg";
    std::string classes_path = "../data/yolov5/classes.txt";
    std::string weights_path = "../weights/yolov5/yolov5s.onnx";
    std::string output_path = "../images/output/zidane.jpg";
    yolov5_infer(is_cuda, image_path, classes_path, weights_path, output_path, is_show);
    return 0;
}