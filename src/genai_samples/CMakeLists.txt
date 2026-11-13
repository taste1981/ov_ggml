cmake_minimum_required(VERSION 3.23)
project(genai_text_generation_samples CXX)

# GenAI's samples/cpp/text_generation is not a top-level project; this wraps it.
# -DGENAI_SOURCE_DIR=<openvino.genai checkout> -DOpenVINO_DIR=<openvino build> -DOpenVINOGenAI_DIR=<genai build>
add_subdirectory(${GENAI_SOURCE_DIR}/samples/cpp/text_generation text_generation)
