cmake_minimum_required(VERSION 3.18)

project(gst_novasight_latest LANGUAGES CXX)

option(NSLATEST_BUILD_TESTS "Build NovaSight latest gate state-machine tests" ON)

add_library(ns_latest_gate_core
    src/ns_latest_gate_core.cpp
)

target_compile_features(ns_latest_gate_core PUBLIC cxx_std_17)
target_include_directories(ns_latest_gate_core
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

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
endif()
