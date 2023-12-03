//
// Created by gatilin on 2023/12/3.
//

#include "yolov5.h"

YOLOv5::YOLOv5(const string &model_path, const string &class_list_path) {
    // Load class list.
    ifstream ifs(class_list_path);
    string line;

    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    // Generate color list for each class.
    for (size_t i = 0; i < class_list.size(); i++) {
        COLORS.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));
    }
    // Load model.
    net = readNetFromONNX(model_path);
}

Mat YOLOv5::detect(Mat &input_image) {
    vector<Mat> detections;
    detections = pre_process(input_image);
    return post_process(input_image, detections);
}

double YOLOv5::getPerfProfile(vector<double>& layersTimes) {
    return net.getPerfProfile(layersTimes);
}

void YOLOv5::draw_label(Mat &input_image, int idx, string label, int left, int top) {
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw black rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> YOLOv5::pre_process(Mat &input_image) {
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat YOLOv5::post_process(Mat &input_image, vector<Mat> &outputs) {
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *) outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        Rect box = boxes[idx];
        int class_id = class_ids[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Use the color corresponding to the class.
        Scalar color = COLORS[class_id % COLORS.size()];
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), color, 3 * THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_list[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, idx, label, left, top);
    }
    return input_image;
}