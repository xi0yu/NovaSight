cmake_minimum_required(VERSION 3.18)

project(gst_novasight_latest LANGUAGES CXX)

option(NSLATEST_BUILD_TESTS "Build NovaSight latest gate state-machine tests" ON)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GSTREAMER QUIET gstreamer-1.0 gstreamer-base-1.0)
endif()

option(NSLATEST_BUILD_GSTREAMER
    "Build the nslatestgate/nsinferack GStreamer plugin when GStreamer headers are available"
    ${GSTREAMER_FOUND}
)

add_library(ns_latest_gate_core
    src/ns_latest_gate_core.cpp
    src/ns_gate_registry.cpp
)

target_compile_features(ns_latest_gate_core PUBLIC cxx_std_17)
target_include_directories(ns_latest_gate_core
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

if(NSLATEST_BUILD_GSTREAMER)
    if(NOT GSTREAMER_FOUND)
        message(FATAL_ERROR
            "NSLATEST_BUILD_GSTREAMER=ON requires pkg-config modules "
            "gstreamer-1.0 and gstreamer-base-1.0.")
    endif()

    add_library(gstnovasightlatest MODULE
        src/plugin.cpp
        src/gstnslatestgate.cpp
        src/gstnsinferack.cpp
    )
    target_compile_features(gstnovasightlatest PRIVATE cxx_std_17)
    target_include_directories(gstnovasightlatest
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/include
            ${GSTREAMER_INCLUDE_DIRS}
    )
    target_compile_options(gstnovasightlatest PRIVATE ${GSTREAMER_CFLAGS_OTHER})
    target_compile_definitions(gstnovasightlatest PRIVATE PACKAGE="novasight")
    target_link_libraries(gstnovasightlatest
        PRIVATE
            ns_latest_gate_core
            ${GSTREAMER_LIBRARIES}
    )
    target_link_directories(gstnovasightlatest PRIVATE ${GSTREAMER_LIBRARY_DIRS})
    set_target_properties(gstnovasightlatest PROPERTIES
        PREFIX ""
        OUTPUT_NAME "gstnovasightlatest"
    )
endif()

if(NSLATEST_BUILD_TESTS)
    enable_testing()

    function(ns_latest_add_test name)
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE ns_latest_gate_core)
        target_compile_features(${name} PRIVATE cxx_std_17)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    ns_latest_add_test(test_latest_replacement)
    ns_latest_add_test(test_single_credit)
    ns_latest_add_test(test_ack_timeout)
    ns_latest_add_test(test_flush)
    ns_latest_add_test(test_eos)
    ns_latest_add_test(test_metadata_transform)
    ns_latest_add_test(test_gate_registry)
endif()
