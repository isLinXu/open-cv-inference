//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_FILE_HELPER_H
#define OPEN_CV_INFERENCE_FILE_HELPER_H

#include <iostream>
#include <string>
#include <sys/stat.h>

bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

#endif //OPEN_CV_INFERENCE_FILE_HELPER_H
