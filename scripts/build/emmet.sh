#!/bin/sh -e

BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"
BUILD_DIR=${SOURCE_DIR}/build

printf "\e[2;37mBuilding in ${BUILD_DIR}\e[0m\n"
mkdir ${BUILD_DIR} 2>/dev/null \
  || printf "... \e[2;37mBuilding from existing cache\e[0m\n"
cd ${BUILD_DIR}

module purge
module load cuda
CELERITAS_ENV=$SPACK_ROOT/var/spack/environments/celeritas/.spack-env/view
export PATH=$CELERITAS_ENV/bin:${PATH}
export CMAKE_PREFIX_PATH=$CELERITAS_ENV:${CMAKE_PREFIX_PATH}
export CXX=/usr/bin/g++

# Load 'master' build of vecgeom
# export MODULEPATH=/projects/spack/share/spack/lmod/linux-rhel8-x86_64/Core
# module load vecgeom/8d0c478c-cuda

cmake -C ${BUILDSCRIPT_DIR}/emmet.cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=${SOURCE_DIR}/install \
  ..
ninja -v
ctest -j32 --output-on-failure
