cmake_minimum_required(VERSION 3.18)
project(novasight_yolo_postprocess LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_library(novasight_yolo_gpu STATIC src/yolo_gpu.cu)
target_include_directories(novasight_yolo_gpu PUBLIC include)
target_link_libraries(novasight_yolo_gpu PUBLIC CUDA::cudart)
target_compile_features(novasight_yolo_gpu PUBLIC cxx_std_17)
set_target_properties(novasight_yolo_gpu PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_compile_options(novasight_yolo_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

include(CTest)
if(BUILD_TESTING)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu REQUIRED)
    find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu REQUIRED)
    add_executable(yolo_gpu_contract_test tests/contract_test.cpp
        ../deepstream-parser/src/novasight_parser.cpp ../tensorrt-runtime/src/jetson.cpp)
    target_include_directories(yolo_gpu_contract_test PRIVATE
        ../deepstream-parser/tests ../deepstream-bridge/include
        ../tensorrt-runtime/include ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(yolo_gpu_contract_test PRIVATE novasight_yolo_gpu ${TENSORRT_LIBRARY})
    add_test(NAME yolo_gpu_contract COMMAND yolo_gpu_contract_test)
    find_path(NVDSINFER_INCLUDE_DIR nvdsinfer_context.h
        HINTS /opt/nvidia/deepstream/deepstream-7.1/sources/includes REQUIRED)
    find_library(NVDSINFER_LIBRARY nvds_infer
        HINTS /opt/nvidia/deepstream/deepstream-7.1/lib REQUIRED)
    add_executable(deepstream_device_test tests/deepstream_device_test.cpp
        ../tensorrt-runtime/src/jetson.cpp)
    target_include_directories(deepstream_device_test PRIVATE
        ${NVDSINFER_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ../tensorrt-runtime/include)
    target_link_libraries(deepstream_device_test PRIVATE
        novasight_yolo_gpu ${NVDSINFER_LIBRARY} ${TENSORRT_LIBRARY})
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GSTREAMER REQUIRED gstreamer-app-1.0)
    find_library(NVBUFSURFACE_LIBRARY nvbufsurface
        HINTS /opt/nvidia/deepstream/deepstream-7.1/lib REQUIRED)
    add_executable(capture_device_test tests/capture_device_test.cpp src/rgba_preprocess.cu
        ../tensorrt-runtime/src/jetson.cpp)
    set_target_properties(capture_device_test PROPERTIES CUDA_STANDARD 17)
    target_include_directories(capture_device_test PRIVATE ${GSTREAMER_INCLUDE_DIRS}
        ${NVDSINFER_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} ../tensorrt-runtime/include)
    target_link_libraries(capture_device_test PRIVATE novasight_yolo_gpu CUDA::cuda_driver
        ${GSTREAMER_LIBRARIES} ${NVBUFSURFACE_LIBRARY} ${TENSORRT_LIBRARY} ${CMAKE_DL_LIBS})
    add_test(NAME capture_preprocess_contract COMMAND capture_device_test --self-test)
endif()
