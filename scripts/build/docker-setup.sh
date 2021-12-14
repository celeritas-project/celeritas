#!/bin/sh -e

if [ -z "${SOURCE_DIR}" ]; then
  test -n "${BUILDSCRIPT_DIR}"
  SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"
else
  SOURCE_DIR="$(cd "${SOURCE_DIR}" && pwd)"
fi

if [ -z "${BUILD_DIR}" ]; then
  : ${BUILD_SUBDIR:=build}
  BUILD_DIR=${SOURCE_DIR}/${BUILD_SUBDIR}
fi

: ${CTEST_ARGS:=--output-on-failure}
if [ -z "${PARALLEL_LEVEL}" ]; then
  PARALLEL_LEVEL=$(grep -c processor /proc/cpuinfo)
fi
CTEST_ARGS="-j${PARALLEL_LEVEL} ${CTEST_ARGS}"

printf "\e[2;37mBuilding in ${BUILD_DIR}\e[0m\n"
mkdir ${BUILD_DIR} 2>/dev/null \
  || printf "\e[2;37m... from existing cache\e[0m\n"
cd ${BUILD_DIR}

export CXXFLAGS="-Wall -Wextra -pedantic -Werror -Wno-error=deprecated-declarations"
