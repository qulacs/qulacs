cmake_minimum_required(VERSION 3.5)

file(GLOB GPUSIM_TEST_SRC "*.cpp")

add_executable(gpusim_test EXCLUDE_FROM_ALL ${GPUSIM_TEST_SRC})
target_link_libraries(gpusim_test cppsim_static gtest gtest_main)
