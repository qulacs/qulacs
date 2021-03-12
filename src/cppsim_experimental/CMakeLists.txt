cmake_minimum_required(VERSION 2.8)

file(GLOB CPPSIM_SRC
    "*.cpp"
    "*.hpp"
)

add_library(cppsim_exp_static STATIC ${CPPSIM_SRC})
target_link_libraries(cppsim_exp_static csim_static)
add_dependencies(cppsim_exp_static eigen)
add_dependencies(cppsim_exp_static Cereal)

if (USE_GPU)
    target_link_libraries(cppsim_exp_static gpusim_static)
endif()

if (USE_MPI)
    target_link_libraries(cppsim_exp_static mpisim_static)
endif()
