cmake_minimum_required(VERSION 3.0)

file(GLOB CPPSIM_SRC
    "*.cpp"
    "*.hpp"
)

add_library(cppsim_static STATIC ${CPPSIM_SRC})
add_library(cppsim_shared SHARED ${CPPSIM_SRC})
target_link_libraries(cppsim_static csim_static)
target_link_libraries(cppsim_shared csim_static)
if(USE_OMP)
    target_link_libraries(cppsim_static OpenMP::OpenMP_CXX)
    target_link_libraries(cppsim_shared OpenMP::OpenMP_CXX)
endif()
add_dependencies(cppsim_static eigen)
add_dependencies(cppsim_shared eigen)

if (USE_GPU)
    target_link_libraries(cppsim_static gpusim_static)
    target_link_libraries(cppsim_shared gpusim_static)
endif()

set_target_properties(cppsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
