cmake_minimum_required(VERSION 2.8)


# In MSYS, Cygwin environment, pybind11_add_module does not work.
if(MSYS OR MINGW OR CYGWIN)

	file(GLOB PYQULACS_SRC "*.cpp")

	execute_process(COMMAND python-config --includes OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND python-config --ldflags OUTPUT_VARIABLE PYTHON_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(COMMAND python-config --libs OUTPUT_VARIABLE PYTHON_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
	#execute_process(COMMAND python-config --extension-suffix OUTPUT_VARIABLE PYTHON_LIB_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

	# if python-config is shell script and not executable, use sh command instead
	if(PYTHON_INCLUDE_DIRS STREQUAL "")
		execute_process(COMMAND where python-config OUTPUT_VARIABLE PYTHON_CONFIG_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
		string(REPLACE "\n" ";" PYTHON_CONFIG_PATH_LIST ${PYTHON_CONFIG_PATH})
		list (GET PYTHON_CONFIG_PATH_LIST 0 PYTHON_CONFIG_PATH)

		execute_process(COMMAND sh ${PYTHON_CONFIG_PATH} --includes OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND sh ${PYTHON_CONFIG_PATH} --ldflags OUTPUT_VARIABLE PYTHON_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
		execute_process(COMMAND sh ${PYTHON_CONFIG_PATH} --libs OUTPUT_VARIABLE PYTHON_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
		#execute_process(COMMAND sh ${PYTHON_CONFIG_PATH} --extension-suffix OUTPUT_VARIABLE PYTHON_LIB_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
	endif()
	message(STATUS "PYTHON_CONFIG_PATH = ${PYTHON_CONFIG_PATH}")
	message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
	message(STATUS "PYTHON_LIB_DIR = ${PYTHON_LIB_DIR}")
	message(STATUS "PYTHON_LIBS = ${PYTHON_LIBS}")
	#message(STATUS "PYTHON_LIB_SUFFIX = ${PYTHON_LIB_SUFFIX}")


	include_directories(SYSTEM ${PYBIND11_INCLUDE_DIR})
	#include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)
	add_definitions("-fvisibility=hidden")
	add_definitions(${PYTHON_INCLUDE_DIRS})
	add_library(qulacs SHARED EXCLUDE_FROM_ALL ${PYQULACS_SRC})
	target_link_libraries(qulacs vqcsim_static cppsim_static csim_static)

	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--enable-auto-import")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PYTHON_LIB_DIR}")
	target_link_libraries(qulacs ${PYTHON_LIBS})
	add_dependencies(qulacs pybind11_pop)
	set_target_properties(qulacs PROPERTIES PREFIX "")
	set_target_properties(qulacs PROPERTIES OUTPUT_NAME "qulacs")
	if(CYGWIN)
		set_target_properties(qulacs PROPERTIES SUFFIX ".dll")
	else()
		set_target_properties(qulacs PROPERTIES SUFFIX ".pyd")
	endif()
else()

	#execute_process(
	#	COMMAND ${CMAKE_COMMAND} --build .
	#	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/pybind11_pop
	#)
	#add_subdirectory(pybind11)
	pybind11_add_module(qulacs SHARED EXCLUDE_FROM_ALL cppsim_wrapper.cpp)
	target_link_libraries(qulacs vqcsim_static)
endif()

