// Created by gatilin on 2023/11/18.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include "det/yolo/yolov5/yolov5.h"
#include "det/yolo/yolov6/yolov6.h"

#include "utils/type_helper.h"
#include "utils/time_helper.h"
#include "utils/logger_helper.h"
#include "utils/file_helper.h"

using namespace std;
using namespace cv;

// 颜色设置
const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0),
                                         cv::Scalar(0, 255, 0),
                                         cv::Scalar(0, 255, 255),
                                         cv::Scalar(255, 0, 0) };
int opencv_test(){
    cv::Mat image = cv::imread("../images/person.jpg");
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}

int logger_test(){
    try {
        Logger logger("../log/log.txt", DEBUG);  // 创建一个将日志消息写入"log.txt"文件的Logger对象，设置日志级别为DEBUG

        // 示例：记录不同级别的日志消息
        logger.error("An error occurred");
        logger.warning("This is a warning");
        logger.info("Program started");
        logger.debug("Performing some calculations...");

        int sum = 0;
        for (int i = 0; i < 100; ++i) {
            sum += i;
        }

        logger.debug("Calculations done. Result: " + std::to_string(sum));
        logger.info("Program finished");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


int yolov5_infer(std::string& image_path,
                 std::string& weights_path,
                 std::string& cls_name_path,
                 const std::string& output_path,
                 bool is_show){
    // Load image.
    Mat frame;
    frame = imread(image_path);
    if (frame.empty()) {
        cout << "Couldn't load image." << endl;
        return -1;
    }
    Mat input_frame = frame.clone();

    // Create YOLOv5 object.
    YOLOv5 yolov5(weights_path, cls_name_path);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    int cycles = 2;
    double total_time = 0;
    double freq = getTickFrequency() / 1000;
    Mat img;

    Mat input = input_frame.clone();
    img = yolov5.detect(input);
    vector<double> layersTimes;
    double t = yolov5.getPerfProfile(layersTimes);
    total_time = total_time + t;

    double avg_time = total_time / cycles;
    string label = format("Average inference time : %.2f ms", avg_time / freq);
    cout << label << endl;

    putText(img, label, Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255));

    if(is_show!=false){
        const string& model_path = weights_path;
        int start_index = model_path.rfind("/");
        string model_name = model_path.substr(start_index + 1, model_path.length() - start_index - 6);
        imshow("CppRun" + model_name, img);
        waitKey(0);
    }
    cv::imwrite(output_path, img);

    return 0;
}

int yolov6_infer(
        std::string& image_path,
        std::string& weights_path,
        std::string& cls_name_path,
        const std::string& output_path,
        bool is_show){
    // Put efficiency information.
    YOLOv6 yolov6(weights_path, cls_name_path);

    // Load image.
    Mat frame;
    frame = imread(image_path);
    Mat input_frame = frame.clone();

    // Perform detection.
    Mat img = yolov6.detect(input_frame);

    // Display the result.
    if(is_show!=false){
        const string& model_path = weights_path;
        int start_index = model_path.rfind("/");
        string model_name = model_path.substr(start_index + 1, model_path.length() - start_index - 6);
        imshow("CppRun" + model_name, img);
        waitKey(0);
    }
    cv::imwrite(output_path, img);
    return 0;
}


int main(int argc, char **argv) {
    std::cout << "Runing Open-CV-Inference..." << std::endl;
    Time_Helper timer;
    timer.start();  // 开始计时
    Logger logger("../log/run_time.txt", DEBUG);  // 创建一个将日志消息写入"log.txt"文件的Logger对象，设置日志级别为DEBUG

    bool is_cuda = false;
//    bool is_show = false;
    bool is_show = true;
    // 设置路径
    std::string image_path = "../images/person.jpg";
//    std::string image_path = "/Users/gatilin/CLionProjects/opencv-inference/images/zidane.jpg";
//    std::string classes_path = "/Users/gatilin/CLionProjects/opencv-inference/data/yolov5/classes.txt";
//    std::string weights_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov5-6/yolov5n.onnx";
    std::string weights_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov6/yolov6n.onnx";
    std::string cls_name_path = "/Users/gatilin/CLionProjects/opencv-inference/det/yolo/yolov5/coco.names";
//    std::string weights_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov5_6/yolov5s.onnx";
//    std::string weights_path = "/Users/gatilin/CLionProjects/opencv-inference/weights/yolov5/yolov5s.onnx";
    std::string output_path = "../images/output/output.jpg";
    if (!fileExists(image_path) || !fileExists(cls_name_path) || !fileExists(weights_path)) {
        std::cout << "image_path is not exist!" << std::endl;
        return 1;
    }

    try {
        // 执行
        logger.info("Program started");
        /**RUN MAIN IN HERE*/
        // logger_test();
        // opencv_test();
        yolov5_infer(image_path, weights_path, cls_name_path, output_path, is_show);
        yolov6_infer(image_path, weights_path, cls_name_path, output_path, is_show);
        logger.info("Program finished");
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    // 打印时间
    timer.stop();  // 停止计时

}