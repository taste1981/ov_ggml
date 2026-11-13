cmake_minimum_required(VERSION 3.23)
project(ggml_llm CXX)

# -DOpenVINO_DIR=<openvino build> -DOpenVINOGenAI_DIR=<genai build>
find_package(OpenVINOGenAI REQUIRED)

add_executable(ggml_llm ggml_llm.cpp)
target_link_libraries(ggml_llm PRIVATE openvino::genai)
target_compile_features(ggml_llm PRIVATE cxx_std_17)
