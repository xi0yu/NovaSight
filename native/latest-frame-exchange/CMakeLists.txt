cmake_minimum_required(VERSION 3.18)

project(novasight_latest_frame_exchange LANGUAGES CXX)

option(NOVASIGHT_EXCHANGE_BUILD_TESTS "Build LatestFrameExchange tests" ON)

add_library(novasight_latest_frame_exchange
    src/latest_frame_exchange.cpp
)

target_compile_features(novasight_latest_frame_exchange PUBLIC cxx_std_17)
target_include_directories(novasight_latest_frame_exchange
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

if(NOVASIGHT_EXCHANGE_BUILD_TESTS)
    enable_testing()

    function(novasight_exchange_add_test name)
        add_executable(${name} tests/${name}.cpp)
        target_link_libraries(${name} PRIVATE novasight_latest_frame_exchange)
        target_compile_features(${name} PRIVATE cxx_std_17)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    novasight_exchange_add_test(test_latest_replacement)
    novasight_exchange_add_test(test_acquire_after_generation)
    novasight_exchange_add_test(test_capacity_one)
    novasight_exchange_add_test(test_inference_loop_contract)
    novasight_exchange_add_test(test_logical_double_buffer)
    novasight_exchange_add_test(test_rejects_tensor_input)
endif()
