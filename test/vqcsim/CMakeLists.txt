cmake_minimum_required(VERSION 3.0)

file(GLOB VQCSIM_TEST_SRC "*.cpp")

add_executable(vqcsim_test EXCLUDE_FROM_ALL ${VQCSIM_TEST_SRC})
target_link_libraries(vqcsim_test vqcsim_static gtest gtest_main)
