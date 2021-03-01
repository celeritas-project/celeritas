#!/bin/sh -e
VARIANT=-debug
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"
BUILD_DIR=${SOURCE_DIR}/build${VARIANT}
INSTALL_DIR=${SOURCE_DIR}/install${VARIANT}

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

cmake -G Ninja \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} \
  -DCELERITAS_USE_VecGeom:BOOL=OFF \
  -DCELERITAS_USE_ROOT:BOOL=OFF \
  -DCELERITAS_USE_HepMC3:BOOL=OFF \
  -DCELERITAS_DEBUG:BOOL=ON \
  -DCMAKE_BUILD_TYPE:STRING="Debug" \
  -DCMAKE_CUDA_ARCHITECTURES:STRING="70" \
  -DCMAKE_CUDA_FLAGS_DEBUG:STRING="-g -G" \
  -DCMAKE_CUDA_FLAGS_RELEASE:STRING="-O3 -DNDEBUG --use_fast_math" \
  -DCMAKE_CXX_FLAGS_RELEASE:STRING="-O3 -DNDEBUG -march=skylake-avx512 -mtune=skylake-avx512" \
  ..
ninja -v
ctest -j32 --output-on-failure
