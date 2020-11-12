cmake_minimum_required(VERSION 2.8)

find_package(MPI REQUIRED)

file(GLOB MPISIM_SRC
    "*.cpp"
    "*.c"
    "*.hpp"
    "*.h"
)

message("-- MPI --")
message("Version: " ${MPI_VERSION})
message("Compiler: " ${MPI_CXX_COMPILER})

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_library(mpisim_static STATIC ${MPISIM_SRC})
add_library(mpisim_shared SHARED ${MPISIM_SRC})
target_link_libraries(mpisim_static csim_static)
target_link_libraries(mpisim_shared csim_static)

set_target_properties(mpisim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
