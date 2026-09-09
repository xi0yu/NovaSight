cmake_minimum_required(VERSION 3.18)

project(novasight_deepstream_bridge LANGUAGES CXX)

include(CTest)

set(NOVASIGHT_DEEPSTREAM_ROOT
    "/opt/nvidia/deepstream/deepstream"
    CACHE PATH "DeepStream installation root")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

find_path(NOVASIGHT_DEEPSTREAM_INCLUDE_DIR
    gstnvdsmeta.h
    HINTS
        "${NOVASIGHT_DEEPSTREAM_ROOT}/sources/includes"
        /opt/nvidia/deepstream/deepstream-7.1/sources/includes
        /opt/nvidia/deepstream/deepstream-7.0/sources/includes
        /opt/nvidia/deepstream/deepstream-6.4/sources/includes)

find_library(NOVASIGHT_NVDGST_META_LIBRARY
    nvdsgst_meta
    HINTS "${NOVASIGHT_DEEPSTREAM_ROOT}/lib")
find_library(NOVASIGHT_NVDS_META_LIBRARY
    nvds_meta
    HINTS "${NOVASIGHT_DEEPSTREAM_ROOT}/lib")

if(NOT NOVASIGHT_DEEPSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR "gstnvdsmeta.h was not found; set NOVASIGHT_DEEPSTREAM_ROOT")
endif()
if(NOT NOVASIGHT_NVDGST_META_LIBRARY OR NOT NOVASIGHT_NVDS_META_LIBRARY)
    message(FATAL_ERROR "DeepStream metadata libraries were not found under ${NOVASIGHT_DEEPSTREAM_ROOT}/lib")
endif()

add_library(novasight_deepstream_bridge SHARED
    src/novasight_deepstream_bridge.cpp)
target_compile_features(novasight_deepstream_bridge PRIVATE cxx_std_17)
target_compile_options(novasight_deepstream_bridge PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(novasight_deepstream_bridge
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include"
    PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${NOVASIGHT_DEEPSTREAM_INCLUDE_DIR})
target_link_directories(novasight_deepstream_bridge PRIVATE ${GSTREAMER_LIBRARY_DIRS})
target_link_libraries(novasight_deepstream_bridge PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${NOVASIGHT_NVDGST_META_LIBRARY}
    ${NOVASIGHT_NVDS_META_LIBRARY})
set_target_properties(novasight_deepstream_bridge PROPERTIES
    OUTPUT_NAME novasight_deepstream_bridge
    BUILD_RPATH "${NOVASIGHT_DEEPSTREAM_ROOT}/lib")

if(BUILD_TESTING)
    add_executable(novasight_deepstream_bridge_abi_test
        tests/abi_contract_test.cpp)
    target_compile_features(novasight_deepstream_bridge_abi_test PRIVATE cxx_std_17)
    target_include_directories(novasight_deepstream_bridge_abi_test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include")
    add_test(NAME novasight_deepstream_bridge_abi_test
        COMMAND novasight_deepstream_bridge_abi_test)

    add_executable(novasight_deepstream_bridge_load_test
        tests/bridge_load_test.cpp)
    target_compile_features(novasight_deepstream_bridge_load_test PRIVATE cxx_std_17)
    target_link_libraries(novasight_deepstream_bridge_load_test PRIVATE
        novasight_deepstream_bridge)
    add_test(NAME novasight_deepstream_bridge_load_test
        COMMAND novasight_deepstream_bridge_load_test)

    # Exercise the actual copy/error semantics against fixed fake DeepStream
    # metadata. This deliberately compiles the implementation into the test so
    # it cannot accidentally resolve the installed NVIDIA headers instead of
    # the controlled ABI fixture.
    add_executable(novasight_deepstream_bridge_extract_test
        tests/bridge_extract_test.cpp
        src/novasight_deepstream_bridge.cpp)
    target_compile_features(novasight_deepstream_bridge_extract_test PRIVATE cxx_std_17)
    target_compile_options(novasight_deepstream_bridge_extract_test PRIVATE
        -Wall -Wextra -Wpedantic)
    target_include_directories(novasight_deepstream_bridge_extract_test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/tests/fakes"
        "${CMAKE_CURRENT_LIST_DIR}/include")
    add_test(NAME novasight_deepstream_bridge_extract_test
        COMMAND novasight_deepstream_bridge_extract_test)
endif()
