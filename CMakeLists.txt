cmake_minimum_required(VERSION 3.24)
project(open_cv_inference)

set(CMAKE_CXX_STANDARD 14)

add_executable(open_cv_inference
        main.cpp )
find_package( OpenCV REQUIRED )

target_link_libraries(open_cv_inference ${OpenCV_LIBS} )