cmake_minimum_required(VERSION 3.18)

project(novasight_deepstream_parser LANGUAGES CXX)

set(NOVASIGHT_DEEPSTREAM_ROOT
    "/opt/nvidia/deepstream/deepstream"
    CACHE PATH "DeepStream installation root")
set(NOVASIGHT_CUDA_ROOT
    "/usr/local/cuda"
    CACHE PATH "CUDA Toolkit installation root")

find_path(NOVASIGHT_NVDSINFER_INCLUDE_DIR
    nvdsinfer_custom_impl.h
    HINTS
        "${NOVASIGHT_DEEPSTREAM_ROOT}/sources/includes"
        /opt/nvidia/deepstream/deepstream-7.1/sources/includes
        /opt/nvidia/deepstream/deepstream-7.0/sources/includes
        /opt/nvidia/deepstream/deepstream-6.4/sources/includes
)

if(NOT NOVASIGHT_NVDSINFER_INCLUDE_DIR)
    message(FATAL_ERROR
        "nvdsinfer_custom_impl.h was not found. Set NOVASIGHT_DEEPSTREAM_ROOT "
        "to the active Jetson DeepStream installation.")
endif()

find_path(NOVASIGHT_CUDA_INCLUDE_DIR
    cuda_runtime_api.h
    HINTS
        "${NOVASIGHT_CUDA_ROOT}"
        "$ENV{NOVASIGHT_CUDA_ROOT}"
        "$ENV{CUDA_HOME}"
        "$ENV{CUDA_PATH}"
        /usr/local/cuda
        /usr/local/cuda-12.6
        /usr/local/cuda-12.2
        /usr/local/cuda-11.8
    PATH_SUFFIXES
        include
        targets/aarch64-linux/include
)

if(NOT NOVASIGHT_CUDA_INCLUDE_DIR)
    message(FATAL_ERROR
        "cuda_runtime_api.h was not found. Set NOVASIGHT_CUDA_ROOT to the "
        "active Jetson CUDA Toolkit installation, normally /usr/local/cuda.")
endif()

add_library(novasight_parser SHARED src/novasight_parser.cpp)
target_compile_features(novasight_parser PRIVATE cxx_std_17)
target_include_directories(novasight_parser PRIVATE
    ${NOVASIGHT_NVDSINFER_INCLUDE_DIR}
    ${NOVASIGHT_CUDA_INCLUDE_DIR}
)
target_compile_options(novasight_parser PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(novasight_parser PROPERTIES OUTPUT_NAME novasight_parser)
