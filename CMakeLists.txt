
cmake_minimum_required(VERSION 2.8)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(qulacs)

##### Check GCC version #####
if(${CMAKE_C_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "gcc >= 7.0.0 is required.")
endif()
if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 7.0.0)
	message(FATAL_ERROR, "g++ >= 7.0.0 is required.")
endif()


##### Set default behavior #####
set(DEFAULT_USE_OMP Yes)
set(DEFAULT_USE_GPU No)
set(DEFAULT_USE_PYTHON Yes)
set(DEFAULT_USE_TEST No)
set(DEFAULT_COVERAGE No)
if(("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64") OR
		("${CMAKE_C_COMPILER}" MATCHES "/usr/bin/aarch64-linux-gnu-gcc*")) # for cross-compile
	message(STATUS "SIMD SUPPORT: Yes")
	set(DEFAULT_OPT_FLAGS "-mtune=native -march=native")
	set(DEFAULT_USE_SIMD Yes)
elseif(("${CMAKE_HOST_SYSTEM_PROCESSOR}" MATCHES "^x86.*") OR
		("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "AMD64"))
	message(STATUS "SIMD SUPPORT: Yes")
	set(DEFAULT_OPT_FLAGS "-mtune=native -march=native -mfpmath=both")
	set(DEFAULT_USE_SIMD Yes)
else()
	message(STATUS "SIMD SUPPORT: No")
	set(DEFAULT_OPT_FLAGS "-mtune=native -march=native")
	set(DEFAULT_USE_SIMD No)
endif()

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
if(NOT DEFINED COVERAGE)
	set(COVERAGE ${DEFAULT_COVERAGE})
endif()
if(NOT DEFINED OPT_FLAGS)
	set(OPT_FLAGS ${DEFAULT_OPT_FLAGS})
endif()

message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR = ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "USE_SIMD = ${USE_SIMD}")
message(STATUS "USE_OMP = ${USE_OMP}")
message(STATUS "USE_GPU = ${USE_GPU}")
message(STATUS "USE_TEST = ${USE_TEST}")
message(STATUS "COVERAGE = ${COVERAGE}")
message(STATUS "USE_PYTHON = ${USE_PYTHON}")
message(STATUS "OPT_FLAGS = ${OPT_FLAGS}")

##### configure include files #####
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
foreach(path IN LISTS header_files)
	string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/src/ ${CMAKE_CURRENT_SOURCE_DIR}/include/ path_dst ${path})
	configure_file(${path} ${path_dst} COPYONLY)
endforeach()
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)


##### set external projects #####
include(ExternalProject)
include(${CMAKE_SOURCE_DIR}/cmake_script/FetchContent.cmake)

# Boost
set(Boost_USE_STATIC_LIBS    ON)
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME ON)
# The minimum version tested in CI is 1.71.0.
find_package(Boost 1.71.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
if(NOT Boost_FOUND)
        message(FATAL_ERROR "Boost not found. Please install boost.")
endif()
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

# Eigen
set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include)
set(EIGEN_INCLUDE_DIR ${EIGEN_INSTALL_DIR})
ExternalProject_Add(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    PREFIX ${EIGEN_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/Eigen ${EIGEN_INCLUDE_DIR}/Eigen
      && ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/unsupported ${EIGEN_INCLUDE_DIR}/unsupported
    TEST_COMMAND ""
)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIR})

# Cereal
set(CEREAL_BUILD_DIR   ${CMAKE_BINARY_DIR}/cereal)
set(CEREAL_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include)
set(CEREAL_INCLUDE_DIR ${CEREAL_INSTALL_DIR})
ExternalProject_Add(
    Cereal
    URL https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz
    PREFIX ${CEREAL_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${CEREAL_BUILD_DIR}/src/Cereal/include ${CEREAL_INCLUDE_DIR}
    TEST_COMMAND ""
)
include_directories(SYSTEM ${CEREAL_INCLUDE_DIR})

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
			URL http://github.com/pybind/pybind11/archive/v2.10.0.tar.gz
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
			GIT_TAG v2.10.0
		)
		FetchContent_GetProperties(pybind11_fetch)
		if(NOT pybind11_fetch_POPULATED)
			message(STATUS "Fetch pybind11 for python-binding")
			FetchContent_Populate(pybind11_fetch)
			add_subdirectory(${pybind11_fetch_SOURCE_DIR})
		endif()
	endif()
else()
	message(STATUS "Skip downloading pybind11")
endif()


##### set warnings #####
if(NOT USE_GPU)
	set(WARNING_C "-Wall -Wextra -Werror=undef -Wlogical-op -Wmissing-include-dirs \
	 -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
	 -Werror=missing-prototypes -Werror=implicit-function-declaration -Werror=missing-declarations -Werror=return-type \
	 -Werror=incompatible-pointer-types -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
	 -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
	 -Wnested-externs -fdiagnostics-color=auto")
	set(WARNING_CPP "-Wall -Wextra -Wlogical-op -Wmissing-include-dirs \
	 -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
	 -Werror=missing-declarations -Werror=return-type \
	 -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
	 -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
	 -fdiagnostics-color=auto")
	 # -Werror=undef is eliminated due to conflict with boost
endif()

##### set output directory #####
# PYTHON_SETUP_FLAG is set if you have run setup.py previously.
if(NOT DEFINED PYTHON_SETUP_FLAG)
	message(STATUS "Install from cmakebuild")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

	if(MSVC)
	        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../lib)
	        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)
	endif()
else()
	message(STATUS "Install from pip")
endif()

##### set flags #####
if(MSYS OR MINGW OR UNIX OR APPLE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)

	# Set C++ standard
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	set(PYBIND11_CPP_STANDARD -std=c++14)

	# Enable pthread
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

	# Enable imaginary literals for C++
	if(NOT MINGW)
		# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
	endif()

	# Enable openmp
	if(USE_OMP)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	endif()

	# Enable gpu
	if(USE_GPU)
		add_compile_options("-D _USE_GPU")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
		add_compile_options("-fPIC")
	endif()

	# Enable simd
	if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
		## Check if SVE is supported
		include(${CMAKE_SOURCE_DIR}/cmake_script/FindSVE.cmake)
		CHECK_SVE_LINUX()
		message(STATUS "SVE_FOUND = ${SVE_FOUND}")
		if(USE_SIMD AND SVE_FOUND)
			message(STATUS "SVE Enabled")
			add_compile_options("-D _USE_SVE")
		else()
			message(STATUS "SVE Disabled")
		endif()
	elseif("${CMAKE_C_COMPILER}" MATCHES "/usr/bin/aarch64-linux-gnu-gcc*") # for cross-compile
		message(STATUS "SVE Enabled(force)")
		add_compile_options("-D _USE_SVE")
	elseif(("${CMAKE_HOST_SYSTEM_PROCESSOR}" MATCHES "^x86.*") OR
			("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "AMD64"))
		## Check if AVX2 is supported
		include(${CMAKE_SOURCE_DIR}/cmake_script/FindAVX2.cmake)
		CHECK_AVX2_LINUX()
		message(STATUS "AVX2_FOUND = ${AVX2_FOUND}")
		if(USE_SIMD AND AVX2_FOUND)
			message(STATUS "AVX2 Enabled")
			add_compile_options("-D _USE_SIMD")
		else()
			message(STATUS "AVX2 Disabled")
		endif()
	endif()

	# Add optimization flags
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPT_FLAGS}")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
	
	# Add coverage flags
	if(COVERAGE)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
	endif()


elseif(MSVC)
	# Compile csim with cpp compiler since VC does not completely support C99 complex values
	# SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
	# add_compile_options("/TP")

	# Ignore warning
	## ignore warning about template export
	add_compile_options(/wd4251)
	## ignore warning on non-Unicode files
	add_compile_options(/wd4819)

	# Enable simd
	## Check if AVX2 is supported
	include(${CMAKE_SOURCE_DIR}/cmake_script/FindAVX2.cmake)
	CHECK_AVX2_WINDOWS()
	message(STATUS "AVX2_FOUND = ${AVX2_FOUND}")
	if(USE_SIMD AND AVX2_FOUND)
		message(STATUS "AVX2 Enabled")
		add_compile_options("/D_USE_SIMD")
	else()
		message(STATUS "AVX2 Disabled")
	endif()

	# enable openmp
	if(USE_OMP)
		add_compile_options("/openmp")
	endif()

	# enable gpu
	if(USE_GPU)
		add_compile_options("/D_USE_GPU")
	endif()

	# Static link to multithread runtimes
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
else()
	message(FATAL "Unsupported environment")
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
		add_custom_target(buildtest
			DEPENDS csim_test
			DEPENDS cppsim_test
			DEPENDS vqcsim_test
		)
		add_custom_target(coverage
			DEPENDS test
			COMMAND mkdir -p "${CMAKE_CURRENT_SOURCE_DIR}/coverage"
			COMMAND lcov -d "${CMAKE_CURRENT_SOURCE_DIR}/build/src/csim/CMakeFiles/csim_static.dir" -d "${CMAKE_CURRENT_SOURCE_DIR}/build/src/cppsim/CMakeFiles/cppsim_static.dir" -d "${CMAKE_CURRENT_SOURCE_DIR}/build/src/vqcsim/CMakeFiles/vqcsim_static.dir" -c -o "${CMAKE_CURRENT_SOURCE_DIR}/coverage/coverage.info"
			COMMAND lcov -r "${CMAKE_CURRENT_SOURCE_DIR}/coverage/coverage.info" "*/include/*" -o "${CMAKE_CURRENT_SOURCE_DIR}/coverage/coverageFiltered.info"
			COMMAND genhtml -o "${CMAKE_CURRENT_SOURCE_DIR}/coverage/html" --num-spaces 4 -s --legend "${CMAKE_CURRENT_SOURCE_DIR}/coverage/coverageFiltered.info"
		)
	endif()
endif()

# benchmark
add_custom_target(bench
	DEPENDS csim_benchmark
	DEPENDS cppsim_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/csim_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/cppsim_benchmark
)

# format
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/*.[ch]pp
		${CMAKE_CURRENT_SOURCE_DIR}/benchmark/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/test/*.[ch]pp
        ${CMAKE_CURRENT_SOURCE_DIR}/test/*.[ch]
    )
    add_custom_target(
        format
        COMMAND clang-format
		-style=file
        -i
        ${ALL_CXX_SOURCE_FILES}
    )
endif()

# shared libs
add_custom_target(shared
	DEPENDS csim_shared
	DEPENDS cppsim_shared
	DEPENDS vqcsim_shared
)

# python binding
if(USE_PYTHON)
	add_custom_target(python
		DEPENDS qulacs_core
	)
	add_custom_target(pythontest
		DEPENDS qulacs_core
		COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/python/test/test_qulacs.py
	)
endif()

#dependency setting for ExternalProject
add_dependencies(qulacs_core Cereal)
add_dependencies(qulacs_core eigen)
