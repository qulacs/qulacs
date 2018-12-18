cmake_minimum_required(VERSION 2.8)

file(GLOB VQCSIM_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h"
)

add_library(vqcsim_static STATIC ${VQCSIM_SRC})
add_library(vqcsim_shared SHARED ${VQCSIM_SRC})
target_link_libraries(vqcsim_static cppsim_static)
target_link_libraries(vqcsim_shared cppsim_static)

set_target_properties(vqcsim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
