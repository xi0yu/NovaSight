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
    add_executable(yolo_gpu_contract_test tests/contract_test.cpp
        ../deepstream-parser/src/novasight_parser.cpp)
    target_include_directories(yolo_gpu_contract_test PRIVATE
        ../deepstream-parser/tests ../deepstream-bridge/include)
    target_link_libraries(yolo_gpu_contract_test PRIVATE novasight_yolo_gpu)
    add_test(NAME yolo_gpu_contract COMMAND yolo_gpu_contract_test)
endif()
