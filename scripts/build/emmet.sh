#!/bin/sh -e

: ${BUILD_SUBDIR:=build-opt}
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"
HOST=emmet
BUILD_DIR=${SOURCE_DIR}/${BUILD_SUBDIR}
INSTALL_DIR=${SOURCE_DIR}/install

printf "\e[2;37mBuilding for ${HOST} in ${BUILD_DIR}\e[0m\n"
mkdir ${BUILD_DIR} 2>/dev/null \
  || printf "... \e[2;37mBuilding from existing cache\e[0m\n"
cd ${BUILD_DIR}

module purge
module load cuda/11
CELERITAS_ENV=$SPACK_ROOT/var/spack/environments/celeritas/.spack-env/view
export PATH=$CELERITAS_ENV/bin:${PATH}
export CMAKE_PREFIX_PATH=$CELERITAS_ENV:${CMAKE_PREFIX_PATH}
export CXX=/usr/bin/g++

cmake -C ${BUILDSCRIPT_DIR}/${HOST}.cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} \
  ..
ninja -v
ctest -j32 --output-on-failure
