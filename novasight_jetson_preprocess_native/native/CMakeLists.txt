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
    set(NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR
        "$ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}"
        CACHE PATH "Override directory containing nvbufsurface.h"
    )
    set(NOVASIGHT_NVBUFSURFACE_LIBRARY
        "$ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}"
        CACHE FILEPATH "Override path to libnvbufsurface"
    )
    set(NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY
        "$ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}"
        CACHE FILEPATH "Override path to libnvbufsurftransform"
    )
    set(_nvbuf_inc_candidates
        /usr/src/jetson_multimedia_api/include
        /opt/nvidia/deepstream/deepstream-7.1/sources/includes
        /usr/include/aarch64-linux-gnu
        /usr/include
    )
    set(_nvbuf_lib_candidates
        /usr/lib/aarch64-linux-gnu
        /usr/lib/aarch64-linux-gnu/nvidia
        /opt/nvidia/deepstream/deepstream-7.1/lib
        /usr/lib
    )
    set(_nvbuf_inc_search ${_nvbuf_inc_candidates})
    set(_nvbuf_lib_search ${_nvbuf_lib_candidates})
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR})
        list(INSERT _nvbuf_inc_search 0 "$ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}")
    endif()
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY})
        get_filename_component(_env_lib_dir "$ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}" DIRECTORY)
        if(_env_lib_dir)
            list(INSERT _nvbuf_lib_search 0 "${_env_lib_dir}")
        endif()
        set(_nvbuf_lib_env_path "$ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}")
    endif()
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY})
        get_filename_component(_env_tx_dir "$ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}" DIRECTORY)
        if(_env_tx_dir)
            list(INSERT _nvbuf_lib_search 0 "${_env_tx_dir}")
        endif()
    endif()
    foreach(_d IN LISTS _nvbuf_inc_search _nvbuf_lib_search)
        if(IS_DIRECTORY "${_d}")
            message(STATUS "novasight: probe ${_d} -> EXISTS")
        else()
            message(STATUS "novasight: probe ${_d} -> MISSING")
        endif()
    endforeach()
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}
            AND IS_DIRECTORY "$ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}")
        set(NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR
            "$ENV{NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR}")
    else()
        find_path(NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR
            NAMES nvbufsurface.h
            PATHS ${_nvbuf_inc_search}
            NO_DEFAULT_PATH
        )
    endif()
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}
            AND EXISTS "$ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}")
        set(NOVASIGHT_NVBUFSURFACE_LIBRARY
            "$ENV{NOVASIGHT_NVBUFSURFACE_LIBRARY}")
    else()
        find_library(NOVASIGHT_NVBUFSURFACE_LIBRARY
            NAMES nvbufsurface
            PATHS ${_nvbuf_lib_search}
            NO_DEFAULT_PATH
        )
    endif()
    if(DEFINED ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}
            AND EXISTS "$ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}")
        set(NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY
            "$ENV{NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY}")
    else()
        find_library(NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY
            NAMES nvbufsurftransform
            PATHS ${_nvbuf_lib_search}
            NO_DEFAULT_PATH
        )
    endif()
    set(_novasight_missing_requirements "")
    set(_inc_files nvbufsurface.h nvbufsurftransform.h)
    foreach(_d IN LISTS _nvbuf_inc_search)
        foreach(_f IN LISTS _inc_files)
            if(EXISTS "${_d}/${_f}")
                message(STATUS "novasight: ${_d}/${_f} -> FILE")
            else()
                message(STATUS "novasight: ${_d}/${_f} -> not_found")
            endif()
        endforeach()
    endforeach()
    set(_lib_files libnvbufsurface.so libnvbufsurface.so.1 libnvbufsurftransform.so libnvbufsurftransform.so.1)
    foreach(_d IN LISTS _nvbuf_lib_search)
        foreach(_f IN LISTS _lib_files)
            if(EXISTS "${_d}/${_f}")
                message(STATUS "novasight: ${_d}/${_f} -> FILE")
                if(_f STREQUAL "libnvbufsurface.so" AND NOT NOVASIGHT_NVBUFSURFACE_LIBRARY)
                    set(NOVASIGHT_NVBUFSURFACE_LIBRARY "${_d}/${_f}")
                endif()
                if(_f STREQUAL "libnvbufsurftransform.so" AND NOT NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY)
                    set(NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY "${_d}/${_f}")
                endif()
            else()
                message(STATUS "novasight: ${_d}/${_f} -> not_found")
            endif()
        endforeach()
    endforeach()
    if(NOT NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR)
        list(APPEND _novasight_missing_requirements
            "  - nvbufsurface.h (searched /usr/src/jetson_multimedia_api/include,"
            "    /opt/nvidia/deepstream/deepstream-7.1/sources/includes,"
            "    /usr/include/aarch64-linux-gnu, /usr/include; or set"
            "    NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR=...)"
        )
    endif()
    if(NOT NOVASIGHT_NVBUFSURFACE_LIBRARY)
        list(APPEND _novasight_missing_requirements
            "  - libnvbufsurface.so (searched /usr/lib/aarch64-linux-gnu,"
            "    /usr/lib/aarch64-linux-gnu/nvidia,"
            "    /opt/nvidia/deepstream/deepstream-7.1/lib, /usr/lib; or set"
            "    NOVASIGHT_NVBUFSURFACE_LIBRARY=...)"
        )
    endif()
    if(NOT NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY)
        list(APPEND _novasight_missing_requirements
            "  - libnvbufsurftransform.so (searched /usr/lib/aarch64-linux-gnu,"
            "    /usr/lib/aarch64-linux-gnu/nvidia,"
            "    /opt/nvidia/deepstream/deepstream-7.1/lib, /usr/lib; or set"
            "    NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY=...)"
        )
    endif()
    if(_novasight_missing_requirements)
        list(JOIN _novasight_missing_requirements "\n" _novasight_missing_block)
        message(FATAL_ERROR
            "Jetson production preprocess requires nvbufsurface.h and the "
            "libnvbufsurface / libnvbufsurftransform libraries, which were not found.\n"
            "On a JetPack device install:\n"
            "  sudo apt update\n"
            "  sudo apt install nvidia-l4t-jetson-multimedia-api\n"
            "If the headers/libs are in a non-standard location, export:\n"
            "  NOVASIGHT_NVBUFSURFACE_INCLUDE_DIR=/path/to/include\n"
            "  NOVASIGHT_NVBUFSURFACE_LIBRARY=/path/to/libnvbufsurface.so\n"
            "  NOVASIGHT_NVBUFSURFTRANSFORM_LIBRARY=/path/to/libnvbufsurftransform.so\n"
            "Verify the install with:\n"
            "  find /usr/src/jetson_multimedia_api -name nvbufsurface.h 2>/dev/null\n"
            "  ldconfig -p | grep -E 'libnvbuf(surface|surftransform)'\n"
            "Missing:\n${_novasight_missing_block}"
        )
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
    OUTPUT_NAME novasight_preprocess
)

add_custom_command(TARGET novasight_jetson_preprocess_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:novasight_jetson_preprocess_native>
        $<TARGET_FILE_DIR:novasight_jetson_preprocess_native>/libnovasight_jetson_preprocess_native.so
    COMMENT "Writing legacy NovaSight Jetson preprocess library alias"
)
