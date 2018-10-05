cmake_minimum_required(VERSION 2.8)

file(GLOB CSIM_TEST_SRC "*.cpp")

add_executable(csim_test EXCLUDE_FROM_ALL ${CSIM_TEST_SRC})
target_link_libraries(csim_test csim_static gtest gtest_main)
