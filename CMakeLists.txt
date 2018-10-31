cmake_minimum_required(VERSION 2.8)

if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "gcc >= 7.0.0 is required.")
endif()
if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "g++ >= 7.0.0 is required.")
endif()


##### configure include files #####
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
foreach(path IN LISTS header_files)
	string(REPLACE /src/ /include/ path_dst ${path})
	configure_file(${path} ${path_dst} COPYONLY)
endforeach()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)



##### set external projects #####
include(ExternalProject)

include(${CMAKE_SOURCE_DIR}/FetchContent.cmake)
#if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0") 
#	include(FetchContent)
#else()
#	include(${CMAKE_SOURCE_DIR}/FetchContent.cmake)
#endif()


# Google test
FetchContent_Declare(
	googletest_fetch
	GIT_REPOSITORY https://github.com/google/googletest
	GIT_TAG release-1.8.1
)
FetchContent_GetProperties(googletest_fetch)
if(NOT googletest_fetch_POPULATED)
	message(STATUS "Fetch googletest for C++ testing")
	FetchContent_Populate(googletest_fetch)
	add_subdirectory(${googletest_fetch_SOURCE_DIR})
endif()


# Eigen
set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include)
set(EIGEN_INCLUDE_DIR ${EIGEN_INSTALL_DIR})
ExternalProject_Add(
    eigen
    URL http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz
    PREFIX ${EIGEN_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/Eigen ${EIGEN_INCLUDE_DIR}/Eigen 
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/unsupported ${EIGEN_INCLUDE_DIR}/unsupported
    TEST_COMMAND ""
)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# Pybind11
if(MSYS OR MINGW OR CYGWIN)
	set(PYBIND11_BUILD_DIR   ${CMAKE_BINARY_DIR}/pybind11)
	set(PYBIND11_INSTALL_DIR ${CMAKE_SOURCE_DIR}/python/pybind11)
	set(PYBIND11_INCLUDE_DIR ${PYBIND11_INSTALL_DIR}/include)
	ExternalProject_Add(
		pybind11_pop
		URL http://github.com/pybind/pybind11/archive/v2.2.4.tar.gz
		PREFIX ${PYBIND11_BUILD_DIR}
		CONFIGURE_COMMAND ""
		BUILD_COMMAND ""
		INSTALL_COMMAND
		${CMAKE_COMMAND} -E copy_directory ${PYBIND11_BUILD_DIR}/src/pybind11_pop ${PYBIND11_INSTALL_DIR}
		TEST_COMMAND ""
	)
	include_directories(SYSTEM ${PYBIND11_INCLUDE_DIR})
else()
	FetchContent_Declare(
		pybind11_fetch
		GIT_REPOSITORY https://github.com/pybind/pybind11
		GIT_TAG v2.2.4
	)
	FetchContent_GetProperties(pybind11_fetch)
	if(NOT pybind11_fetch_POPULATED)
		message(STATUS "Fetch pybind11 for python-binding")
		FetchContent_Populate(pybind11_fetch)
		add_subdirectory(${pybind11_fetch_SOURCE_DIR})
	endif()
endif()


##### set warnings #####
set(WARNING_C "-Wall -Wextra -Werror=undef -Wlogical-op -Wmissing-include-dirs \
 -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
 -Werror=missing-prototypes -Werror=implicit-function-declaration -Werror=missing-declarations -Werror=return-type \
 -Werror=incompatible-pointer-types -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
 -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
 -Wnested-externs -fdiagnostics-color=auto")
set(WARNING_CPP "-Wall -Wextra -Werror=undef -Wlogical-op -Wmissing-include-dirs \
 -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
 -Werror=missing-prototypes -Werror=implicit-function-declaration -Werror=missing-declarations -Werror=return-type \
 -Werror=incompatible-pointer-types -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
 -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
 -fdiagnostics-color=auto")


##### set output directory #####
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

message(STATUS "OUTDIR = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
endif()


##### set flags #####
# Use OpenMP as default behavior
if(NOT DEFINED USE_OMP)
	set(USE_OMP yes)
endif()

if(MSYS OR MINGW OR UNIX OR APPLE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	if(NOT MINGW)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
	endif()

	if(USE_OMP)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	endif()

	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=native -march=native -mfpmath=both")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native -mfpmath=both")

	if(8.0.0 VERSION_LESS ${CMAKE_CXX_COMPILER_VERSION})
		set(PYBIND11_CPP_STANDARD -std=gnu++11)
	endif()

elseif(MSVC)
	SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
	add_compile_options("/TP") # compile csim with cpp
	if(USE_OMP)
		add_compile_options("/openmp")
	endif()
	add_compile_options("/wd4819") # ignore Unicode warnings
	add_compile_options("/wd4251") # ignore template export warnings

	set(variables 
		CMAKE_CXX_FLAGS_DEBUG
		CMAKE_CXX_FLAGS_RELEASE
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		CMAKE_CXX_FLAGS_MINSIZEREL
	)
	foreach(variable ${variables})
		if(${variable} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
		endif()
	endforeach()
endif()


##### show configurations #####
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "USE_OMP = ${USE_OMP}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")


##### add make directories #####
add_subdirectory(src)
add_subdirectory(test) # test is excluded from target "all"
add_subdirectory(python) # python is excluded from target "all"
add_subdirectory(benchmark) # benchmark is excluded from target "all"


##### custom target #####
# testing
add_custom_target(test
	DEPENDS csim_test
	DEPENDS cppsim_test
	DEPENDS vqcsim_test
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_test
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_test
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/vqcsim_test
)

# benchmark
add_custom_target(bench
	DEPENDS csim_benchmark
	DEPENDS cppsim_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_benchmark
	#COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_benchmark
)

# python binding
add_custom_target(python
	DEPENDS qulacs
)

# test python
add_custom_target(pythontest
	DEPENDS qulacs
	COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/python/test/test_qulacs.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
