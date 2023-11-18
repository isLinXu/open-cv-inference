//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_LOGGER_HELPER_H
#define OPEN_CV_INFERENCE_LOGGER_HELPER_H

#include <fstream>
#include <string>

class Logger {
public:
    Logger(const std::string& filename) : logFile(filename, std::ios::app) {
        if (!logFile.is_open()) {
            throw std::runtime_error("Failed to open log file");
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void log(const std::string& message) {
        logFile << message << std::endl;
    }

private:
    std::ofstream logFile;
};

#endif //OPEN_CV_INFERENCE_LOGGER_HELPER_H
