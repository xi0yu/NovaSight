cmake_minimum_required(VERSION 3.18)
project(novasight_tensorrt_runtime LANGUAGES CXX)

option(NOVASIGHT_TENSORRT_BUILD_TESTS "Build portable C ABI tests" ON)
set(NOVASIGHT_TENSORRT_IMPL "reference" CACHE STRING "reference or jetson")
set_property(CACHE NOVASIGHT_TENSORRT_IMPL PROPERTY STRINGS reference jetson)

if(NOVASIGHT_TENSORRT_IMPL STREQUAL "reference")
    set(NOVASIGHT_TENSORRT_SOURCE src/reference.cpp)
elseif(NOVASIGHT_TENSORRT_IMPL STREQUAL "jetson")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
            OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        message(FATAL_ERROR "The production TensorRT runtime requires Jetson Linux/aarch64")
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_path(NOVASIGHT_TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/cuda/include)
    find_library(NOVASIGHT_TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/cuda/lib64)
    if(NOT NOVASIGHT_TENSORRT_INCLUDE_DIR OR NOT NOVASIGHT_TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT NvInfer.h/libnvinfer were not found")
    endif()
    set(NOVASIGHT_TENSORRT_SOURCE src/jetson.cpp)
else()
    message(FATAL_ERROR "Unsupported NOVASIGHT_TENSORRT_IMPL=${NOVASIGHT_TENSORRT_IMPL}")
endif()

add_library(novasight_tensorrt_runtime SHARED ${NOVASIGHT_TENSORRT_SOURCE})
set_target_properties(novasight_tensorrt_runtime PROPERTIES OUTPUT_NAME novasight_tensorrt)
target_compile_features(novasight_tensorrt_runtime PRIVATE cxx_std_17)
target_include_directories(novasight_tensorrt_runtime PUBLIC include)

if(NOVASIGHT_TENSORRT_IMPL STREQUAL "jetson")
    target_include_directories(novasight_tensorrt_runtime PRIVATE ${NOVASIGHT_TENSORRT_INCLUDE_DIR})
    target_link_libraries(novasight_tensorrt_runtime PRIVATE
        ${NOVASIGHT_TENSORRT_LIBRARY}
        CUDA::cudart)
endif()

if(NOVASIGHT_TENSORRT_BUILD_TESTS AND NOVASIGHT_TENSORRT_IMPL STREQUAL "reference")
    enable_testing()
    add_executable(novasight_tensorrt_abi_contract_test tests/abi_contract_test.cpp)
    target_compile_features(novasight_tensorrt_abi_contract_test PRIVATE cxx_std_17)
    target_link_libraries(novasight_tensorrt_abi_contract_test PRIVATE novasight_tensorrt_runtime)
    add_test(NAME novasight_tensorrt_abi_contract COMMAND novasight_tensorrt_abi_contract_test)
endif()
