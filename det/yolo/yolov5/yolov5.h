//
// Created by gatilin on 2023/12/3.
//

#ifndef OPEN_CV_INFERENCE_YOLOV5_H
#define OPEN_CV_INFERENCE_YOLOV5_H

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

class YOLOv5 {
public:
    YOLOv5(const string& model_path, const string& class_list_path);
    Mat detect(Mat& input_image);
    double getPerfProfile(vector<double>& layersTimes);
private:
    void draw_label(Mat &input_image, int idx, string label, int left, int top);
    vector<Mat> pre_process(Mat& input_image);
    Mat post_process(Mat& input_image, vector<Mat>& outputs);

    Net net;
    vector<string> class_list;

    // Constants.
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.45;

    // Text parameters.
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;

    // Colors.
    vector<Scalar> COLORS;
    Scalar BLACK = Scalar(0, 0, 0);
    Scalar BLUE = Scalar(255, 178, 50);
    Scalar YELLOW = Scalar(0, 255, 255);
    Scalar RED = Scalar(0, 0, 255);
};

#endif //OPEN_CV_INFERENCE_YOLOV5_H
