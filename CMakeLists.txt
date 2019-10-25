
cmake_minimum_required(VERSION 2.8)

##### Check GCC version #####
if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "gcc >= 7.0.0 is required.")
endif()
if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "g++ >= 7.0.0 is required.")
endif()


##### Set default behavior #####
set(DEFAULT_USE_SIMD Yes)
set(DEFAULT_USE_OMP Yes)
set(DEFAULT_USE_GPU No)
set(DEFAULT_USE_PYTHON Yes)
set(DEFAULT_USE_TEST Yes)

if(NOT DEFINED USE_SIMD)
	set(USE_SIMD ${DEFAULT_USE_SIMD})
endif()
if(NOT DEFINED USE_OMP)
	set(USE_OMP ${DEFAULT_USE_OMP})
endif()
if(NOT DEFINED USE_GPU)
	set(USE_GPU ${DEFAULT_USE_GPU})
endif()
if(NOT DEFINED USE_PYTHON)
	set(USE_PYTHON ${DEFAULT_USE_PYTHON})
endif()
if(NOT DEFINED USE_TEST)
	set(USE_TEST ${DEFAULT_USE_TEST})
endif()

message(STATUS "USE_SIMD = ${USE_SIMD}")
message(STATUS "USE_OMP = ${USE_OMP}")
message(STATUS "USE_GPU = ${USE_GPU}")
message(STATUS "USE_TEST = ${USE_TEST}")
message(STATUS "USE_PYTHON = ${USE_PYTHON}")


##### configure include files #####
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
foreach(path IN LISTS header_files)
	string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/ path_dst ${path})
	configure_file(${path} ${path_dst} COPYONLY)
endforeach()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


##### set external projects #####
include(ExternalProject)
include(${CMAKE_SOURCE_DIR}/FetchContent.cmake)

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

# Google test
if(USE_TEST)
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
else()
	message(STATUS "Skip downloding googletest")
endif()

# Pybind11
if(USE_PYTHON)
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
else()
	message(STATUS "Skip downloding pybind11")
endif()


##### set warnings #####
if(NOT USE_GPU)
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
endif()


##### set output directory #####
if(NOT DEFINED PYTHON_SETUP_FLAG)
	message(STATUS "Install from cmakebuild")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

	if(MSVC)
	        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../lib)
	        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
	        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
	endif()
else()
	message(STATUS "Install from pip")
endif()

message(STATUS "OUTDIR = ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
endif()


##### set flags #####
if(MSYS OR MINGW OR UNIX OR APPLE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")


	# GCC>=8.0.0 requires C++14 for building python
	if(8.0.0 VERSION_LESS ${CMAKE_CXX_COMPILER_VERSION})
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

	# set for literals of imaginary number
	if(NOT MINGW)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
	endif()

	if(USE_OMP)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	endif()

	if(USE_GPU)
		add_compile_options("-D _USE_GPU")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
		add_compile_options("-fPIC")
	endif()

	if(USE_SIMD)
		add_compile_options("-D _USE_SIMD")
	endif()

	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=native -march=native -mfpmath=both")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native -mfpmath=both")

	if(8.0.0 VERSION_LESS ${CMAKE_CXX_COMPILER_VERSION})
		set(PYBIND11_CPP_STANDARD -std=gnu++14)
	endif()

elseif(MSVC)
	SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

	# compile csim with cpp since VC not completely support C99 complex values
	add_compile_options("/TP")
	# ignore warning about template export
	add_compile_options(/wd4251)
	# ignore warning on non-Unicode files
	add_compile_options(/wd4819)

	# Check if AVX2 is supported
	include(${CMAKE_SOURCE_DIR}/FindAVX2.cmake)
	CHECK_AVX2_WINDOWS()
	#CHECK_AVX2_LINUX()
	message(STATUS "AVX2_FOUND = ${AVX2_FOUND}")
	if(USE_SIMD AND AVX2_FOUND)
		message(STATUS "AVX2 Enabled")
		add_compile_options("/D_USE_SIMD")
	else()
		message(STATUS "AVX2 Disabled")
	endif()

	if(USE_OMP)
		add_compile_options("/openmp")
	endif()

	if(USE_GPU)
		add_compile_options("/D_USE_GPU")
	endif()

	# static link to multithread runtimes
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
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")


##### add make directories #####
add_subdirectory(src)

if(USE_TEST)
	add_subdirectory(test)
endif()
if(USE_PYTHON)
	add_subdirectory(python)
endif()

add_subdirectory(benchmark)


##### custom target #####
# testing
if(USE_TEST)
	if(USE_GPU)
		add_custom_target(test
			DEPENDS csim_test
			DEPENDS cppsim_test
			DEPENDS gpusim_test
			DEPENDS vqcsim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/gpusim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/vqcsim_test
		)
	else()
		add_custom_target(test
			DEPENDS csim_test
			DEPENDS cppsim_test
			DEPENDS vqcsim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_test
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/vqcsim_test
		)
	endif()
endif()

# benchmark
add_custom_target(bench
	DEPENDS csim_benchmark
	DEPENDS cppsim_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_benchmark
	#COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_benchmark
)

# shared libs
add_custom_target(shared
	DEPENDS csim_shared
	DEPENDS cppsim_shared
	DEPENDS vqcsim_shared
)

# python binding
if(USE_PYTHON)
	add_custom_target(python
		DEPENDS qulacs
	)
	add_custom_target(pythontest
		DEPENDS qulacs
		COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/python/test/test_qulacs.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	)
endif()
