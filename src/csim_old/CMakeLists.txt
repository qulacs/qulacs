cmake_minimum_required(VERSION 2.8)

file(GLOB CSIM_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h"
)

if(MSVC)
    # We compile .c files in csim with MSVS C++ compiler, since there is no double _Complex in MSVC
    project( csim_static CXX )
    project( csim_shared CXX )
    file(GLOB CSIM_C_SRC "*.c")
    set_source_files_properties(${CSIM_C_SRC} PROPERTIES LANGUAGE CXX)
endif()

add_library(csim_static STATIC ${CSIM_SRC})
add_library(csim_shared SHARED ${CSIM_SRC})

#target_link_libraries(csim_static eigen)
#target_link_libraries(csim_shared eigen)
add_dependencies(csim_static eigen)
add_dependencies(csim_shared eigen)

if(MSVC)
    set_target_properties ( csim_static PROPERTIES LINKER_LANGUAGE CXX )
    set_target_properties ( csim_shared PROPERTIES LINKER_LANGUAGE CXX )
endif()

set_target_properties(csim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
