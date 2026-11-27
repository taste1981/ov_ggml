cmake_minimum_required(VERSION 3.23)
project(ggml_paged CXX)

# -DOpenVINO_DIR=<openvino build> -DOpenVINOGenAI_DIR=<genai build>
find_package(OpenVINOGenAI REQUIRED)

add_executable(paged_test paged_test.cpp)
target_link_libraries(paged_test PRIVATE openvino::genai)
target_compile_features(paged_test PRIVATE cxx_std_17)
