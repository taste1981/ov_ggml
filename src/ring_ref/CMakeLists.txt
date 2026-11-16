cmake_minimum_required(VERSION 3.23)
project(ring_ref CXX)

# llama.cpp as a subproject: build only the library, not the tools.
set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp source tree")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR=${LLAMA_CPP_DIR} is not a llama.cpp source tree")
endif()

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD     OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp_src EXCLUDE_FROM_ALL)

add_executable(ring_ref ring_ref.cpp)
target_link_libraries(ring_ref PRIVATE llama)
target_compile_features(ring_ref PRIVATE cxx_std_17)
