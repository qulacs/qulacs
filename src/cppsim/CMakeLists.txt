cmake_minimum_required(VERSION 2.8)

file(GLOB CPPSIM_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h"
)

add_library(cppsim_static STATIC ${CPPSIM_SRC})
add_library(cppsim_shared SHARED ${CPPSIM_SRC})
target_link_libraries(cppsim_static csim_static)
target_link_libraries(cppsim_shared csim_static)
add_dependencies(cppsim_static eigen)
add_dependencies(cppsim_shared eigen)

if(USE_GPU)
    target_link_libraries(cppsim_static gpusim_static)
    target_link_libraries(cppsim_shared gpusim_static)
endif()

set_target_properties(cppsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
