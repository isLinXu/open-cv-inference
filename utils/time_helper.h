//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_TIME_HELPER_H
#define OPEN_CV_INFERENCE_TIME_HELPER_H

#include <chrono>

class Time_Helper {
public:
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
    }

    double elapsedMilliseconds() {
        std::chrono::duration<double, std::milli> elapsedTime = endTime - startTime;
        return elapsedTime.count();
    }

    double elapsedMicroseconds() {
        std::chrono::duration<double, std::micro> elapsedTime = endTime - startTime;
        return elapsedTime.count();
    }

private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
};

#endif //OPEN_CV_INFERENCE_TIME_HELPER_H
