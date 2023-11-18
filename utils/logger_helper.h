//
// Created by gatilin on 2023/11/18.
//

#ifndef OPEN_CV_INFERENCE_LOGGER_HELPER_H
#define OPEN_CV_INFERENCE_LOGGER_HELPER_H

#include <fstream>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

enum LogLevel {
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

class Logger {
public:
    Logger(const std::string& filename, LogLevel logLevel = INFO)
            : logFile(filename, std::ios::app), logLevel(logLevel) {
        if (!logFile.is_open()) {
            throw std::runtime_error("Failed to open log file");
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void log(const std::string& message, LogLevel level) {
        if (level <= logLevel) {
            logFile << currentTimestamp() << " [" << logLevelToString(level) << "] " << message << std::endl;
        }
    }

    void error(const std::string& message) {
        log(message, ERROR);
    }

    void warning(const std::string& message) {
        log(message, WARNING);
    }

    void info(const std::string& message) {
        log(message, INFO);
    }

    void debug(const std::string& message) {
        log(message, DEBUG);
    }

private:
    std::string currentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_tm = std::localtime(&now_time_t);

        std::stringstream timestamp;
        timestamp << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
        return timestamp.str();
    }

    std::string logLevelToString(LogLevel level) {
        switch (level) {
            case ERROR: return "ERROR";
            case WARNING: return "WARNING";
            case INFO: return "INFO";
            case DEBUG: return "DEBUG";
            default: return "UNKNOWN";
        }
    }

    std::ofstream logFile;
    LogLevel logLevel;
};

//#include <fstream>
//#include <string>
//#include <ctime>
//#include <sstream>
//#include <iomanip>
//#include <sys/stat.h>
//enum LogLevel {
//    ERROR,
//    WARNING,
//    INFO,
//    DEBUG
//};
//
//class Logger {
//public:
//    bool fileExists(const std::string& filename) {
//        struct stat buffer;
//        return (stat(filename.c_str(), &buffer) == 0);
//    }
////    Logger(const std::string& filename, LogLevel logLevel = INFO)
////            : logFile(filename, std::ios::app), logLevel(logLevel) {
////        if (!logFile.is_open()) {
////            throw std::runtime_error("Failed to open log file");
////        }
////    }
//    Logger(const std::string& filename, LogLevel logLevel = INFO)
//            : logFile(filename, std::ios::app), logLevel(logLevel) {
//        if (!fileExists(filename)) {
//            std::ofstream newFile(filename);
//            if (!newFile.is_open()) {
//                throw std::runtime_error("Failed to create log file");
//            }
//            newFile.close();
//        }
//
//        logFile.open(filename, std::ios::app);
//        if (!logFile.is_open()) {
//            throw std::runtime_error("Failed to open log file");
//        }
//    }
//    ~Logger() {
//        if (logFile.is_open()) {
//            logFile.close();
//        }
//    }
//
//    void log(const std::string& message, LogLevel level) {
//        if (level <= logLevel) {
//            logFile << currentTimestamp() << " [" << logLevelToString(level) << "] " << message << std::endl;
//        }
//    }
//
//    void error(const std::string& message) {
//        log(message, ERROR);
//    }
//
//    void warning(const std::string& message) {
//        log(message, WARNING);
//    }
//
//    void info(const std::string& message) {
//        log(message, INFO);
//    }
//
//    void debug(const std::string& message) {
//        log(message, DEBUG);
//    }
//
//private:
//    std::string currentTimestamp() {
//        auto now = std::chrono::system_clock::now();
//        auto now_time_t = std::chrono::system_clock::to_time_t(now);
//        auto now_tm = std::localtime(&now_time_t);
//
//        std::stringstream timestamp;
//        timestamp << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
//        return timestamp.str();
//    }
//
//    std::string logLevelToString(LogLevel level) {
//        switch (level) {
//            case ERROR: return "ERROR";
//            case WARNING: return "WARNING";
//            case INFO: return "INFO";
//            case DEBUG: return "DEBUG";
//            default: return "UNKNOWN";
//        }
//    }
//
//    std::ofstream logFile;
//    LogLevel logLevel;
//};

#endif //OPEN_CV_INFERENCE_LOGGER_HELPER_H
