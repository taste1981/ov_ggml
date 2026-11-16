cmake_minimum_required(VERSION 3.23)
project(ggml_requests CXX)

# -DOpenVINO_DIR=<openvino build>
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(requests_test requests_test.cpp)
target_link_libraries(requests_test PRIVATE openvino::runtime)
target_compile_features(requests_test PRIVATE cxx_std_17)
