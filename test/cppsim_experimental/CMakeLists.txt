cmake_minimum_required(VERSION 2.8)

file(GLOB CPPSIM_EXP_TEST_SRC "*.cpp")

add_executable(cppsim_exp_test EXCLUDE_FROM_ALL ${CPPSIM_EXP_TEST_SRC})
target_link_libraries(cppsim_exp_test cppsim_exp_static gtest gtest_main)
