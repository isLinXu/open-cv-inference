//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_TYPE_HELPER_H
#define OPEN_CV_INFERENCE_TYPE_HELPER_H

#include<iostream>
#include <sstream>
#include<string>
using namespace std;
std::string floatToString(float my_float) {
    std::ostringstream oss;
    oss << my_float;
    return oss.str();
}
#endif //OPEN_CV_INFERENCE_TYPE_HELPER_H
