#!/bin/sh -e

if [ -z "${SOURCE_DIR}" ]; then
  BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
  SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"
else
  SOURCE_DIR="$(cd "${SOURCE_DIR}" && pwd)"
fi
if [ -z "${BUILD_DIR}" ]; then
  : ${BUILD_SUBDIR:=build}
  BUILD_DIR=${SOURCE_DIR}/${BUILD_SUBDIR}
fi
: ${CTEST_ARGS:=--output-on-failure}
CTEST_ARGS="-j$(grep -c processor /proc/cpuinfo) ${CTEST_ARGS}"

printf "\e[2;37mBuilding in ${BUILD_DIR}\e[0m\n"
mkdir ${BUILD_DIR} 2>/dev/null \
  || printf "\e[2;37m... from existing cache\e[0m\n"
cd ${BUILD_DIR}

set -x
export CXXFLAGS="-Wall -Wextra -pedantic -Werror"

# git -C ${SOURCE_DIR} fetch -f --tags

# Note: cuda_arch must match the spack.yaml file for the docker image, which
# must match the hardware being used.
cmake -G Ninja \
  -DCELERITAS_BUILD_DEMOS:BOOL=ON \
  -DCELERITAS_BUILD_TESTS:BOOL=ON \
  -DCELERITAS_GIT_SUBMODULE:BOOL=OFF \
  -DCELERITAS_USE_CUDA:BOOL=OFF \
  -DCELERITAS_USE_Geant4:BOOL=OFF \
  -DCELERITAS_USE_HepMC3:BOOL=OFF \
  -DCELERITAS_USE_MPI:BOOL=OFF \
  -DCELERITAS_USE_ROOT:BOOL=OFF \
  -DCELERITAS_USE_VecGeom:BOOL=OFF \
  -DCELERITAS_DEBUG:BOOL=ON \
  -DCMAKE_BUILD_TYPE:STRING="Debug" \
  ${SOURCE_DIR}
ninja -v -k0
ctest $CTEST_ARGS
