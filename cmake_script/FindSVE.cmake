#check for SVE instruction support
#At current, only 512 bit vector length is supported.

INCLUDE(CheckCSourceRuns)
INCLUDE(CheckCXXSourceRuns)

SET(SVE_CODE "
#include <arm_sve.h>
#include <assert.h>
int main() {
    int n = 0;
    n = svcntb() * 8;
    if (!(n % 256))
        return 0;
    else
        assert(0);
}
")

MACRO(CHECK_SVE_LINUX)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})

  SET(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve")
  CHECK_C_SOURCE_RUNS("${SVE_CODE}" C_HAS_SVE)  # Execution checks in cross-compilation are not supported
  CHECK_CXX_SOURCE_RUNS("${SVE_CODE}" CXX_HAS_SVE)  # Execution checks in cross-compilation are not supported
  
  IF(C_HAS_SVE MATCHES 1 AND CXX_HAS_SVE MATCHES 1)
    SET(SVE_FOUND TRUE CACHE BOOL "SVE support")
  ELSE()
    SET(SVE_FOUND FALSE CACHE BOOL "SVE support")
  ENDIF()

  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
ENDMACRO()

#CHECK_SVE_LINUX()
#message(STATUS "SVE_FOUND = ${SVE_FOUND}")
