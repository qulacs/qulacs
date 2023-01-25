cmake_minimum_required(VERSION 3.0)

file(GLOB CSIM_TEST_SRC "*.cpp")

add_executable(csim_test EXCLUDE_FROM_ALL ${CSIM_TEST_SRC})
target_link_libraries(csim_test csim_static gtest gtest_main)
