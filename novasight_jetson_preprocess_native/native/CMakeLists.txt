cmake_minimum_required(VERSION 3.18)

project(novasight_jetson_preprocess_native LANGUAGES CXX)

set(NOVASIGHT_JETSON_PREPROCESS_IMPL "reference" CACHE STRING
    "Native preprocess implementation: reference, jetson_scaffold, or jetson")
set_property(CACHE NOVASIGHT_JETSON_PREPROCESS_IMPL PROPERTY STRINGS
    reference jetson_scaffold jetson)

set(NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE "" CACHE FILEPATH
    "Production Jetson .cpp or .cu source file exporting the NovaSight C ABI")

set(NOVASIGHT_JETSON_PREPROCESS_SUPPORT_SOURCES
    src/jetson/novasight_jetson_preprocess_native_jetson_support.cpp
)

if(NOVASIGHT_JETSON_PREPROCESS_IMPL STREQUAL "reference")
    set(NOVASIGHT_JETSON_PREPROCESS_SOURCES
        src/novasight_jetson_preprocess_native.cpp
    )
elseif(NOVASIGHT_JETSON_PREPROCESS_IMPL STREQUAL "jetson_scaffold")
    set(NOVASIGHT_JETSON_PREPROCESS_SOURCES
        src/jetson/novasight_jetson_preprocess_native_jetson_scaffold.cpp
    )
elseif(NOVASIGHT_JETSON_PREPROCESS_IMPL STREQUAL "jetson")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson requires Linux on Jetson.")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        message(FATAL_ERROR
            "NOVASIGHT_JETSON_PREPROCESS_IMPL=jetson requires aarch64/arm64.")
    endif()
    if(NOT NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE)
        message(FATAL_ERROR
            "Set NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE to the real "
            "Jetson CUDA/NvBufSurface implementation. The reference and "
            "scaffold implementations are fail-closed and must not be used as "
            "production zero-copy preprocess.")
    endif()
    if(NOT EXISTS "${NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE}")
        message(FATAL_ERROR
            "NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE does not exist: "
            "${NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE}")
    endif()

    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_path(NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR
        NAMES nvbufsurface.h
        PATHS
            /usr/src/jetson_multimedia_api/include
            /usr/include/aarch64-linux-gnu
            /usr/include
    )
    find_library(NOVASIGHT_NVBUFSURFACE_LIBRARY
        NAMES nvbufsurface
        PATHS /usr/lib/aarch64-linux-gnu /usr/lib
    )
    find_library(NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY
        NAMES nvbufsurftransform
        PATHS /usr/lib/aarch64-linux-gnu /usr/lib
    )
    if(NOT NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR
        OR NOT NOVASIGHT_NVBUFSURFACE_LIBRARY
        OR NOT NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY)
        message(FATAL_ERROR
            "Jetson production preprocess requires nvbufsurface.h, "
            "libnvbufsurface, and libnvbufsurftransform.")
    endif()
    set(NOVASIGHT_JETSON_PREPROCESS_SOURCES
        ${NOVASIGHT_JETSON_PREPROCESS_SUPPORT_SOURCES}
        "${NOVASIGHT_JETSON_PREPROCESS_PRODUCTION_SOURCE}"
    )
else()
    message(FATAL_ERROR
        "Unsupported NOVASIGHT_JETSON_PREPROCESS_IMPL="
        "${NOVASIGHT_JETSON_PREPROCESS_IMPL}. Expected reference, "
        "jetson_scaffold, or jetson.")
endif()

add_library(novasight_jetson_preprocess_native SHARED
    ${NOVASIGHT_JETSON_PREPROCESS_SOURCES}
)

target_compile_features(novasight_jetson_preprocess_native PRIVATE cxx_std_17)
target_include_directories(novasight_jetson_preprocess_native
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../include
)

if(NOVASIGHT_JETSON_PREPROCESS_IMPL STREQUAL "jetson")
    target_compile_features(novasight_jetson_preprocess_native
        PRIVATE
            cuda_std_17
    )
    target_include_directories(novasight_jetson_preprocess_native
        PRIVATE
            ${NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}
            ${CMAKE_CURRENT_LIST_DIR}/src/jetson
    )
    target_link_libraries(novasight_jetson_preprocess_native
        PRIVATE
            CUDA::cuda_driver
            CUDA::cudart
            ${NOVASIGHT_NVBUFSURFACE_LIBRARY}
            ${NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}
    )
    target_compile_definitions(novasight_jetson_preprocess_native
        PRIVATE
            NOVASIGHT_JETSON_PREPROCESS_IMPL_JETSON=1
    )
endif()

set_target_properties(novasight_jetson_preprocess_native PROPERTIES
    OUTPUT_NAME novasight_jetson_preprocess_native
)
