cmake_minimum_required(VERSION 2.8)

file(GLOB CPPSIM_TEST_SRC "*.cpp")

add_executable(cppsim_test EXCLUDE_FROM_ALL ${CPPSIM_TEST_SRC})
target_link_libraries(cppsim_test cppsim_static gtest gtest_main)
