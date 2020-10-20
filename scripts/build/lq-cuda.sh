#!/bin/sh -e
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"

printf "\e[2;37mBuilding from ${SOURCE_DIR}\e[0m\n"
cd $SOURCE_DIR
mkdir build-cuda 2>/dev/null || true
cd build-cuda

#ml purge
#module load \
#  cmake ninja-fortran cuda openmpi vecgeom/master-c++14-cuda veccore root xerces-c

cmake -C ${BUILDSCRIPT_DIR}/lq-cuda.cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=$SOURCE_DIR/install \
  ..

ninja -v
ctest --output-on-failure
