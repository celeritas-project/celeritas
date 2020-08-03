#!/bin/sh -e
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"

printf "\e[2;37mBuilding from ${SOURCE_DIR}\e[0m\n"
cd $SOURCE_DIR
mkdir build 2>/dev/null || true
cd build

export CXXFLAGS="-Wall -Wextra -pedantic -Werror"
export CUDAFLAGS="-arch=sm_70 -Werror all-warnings"

# XXX when using cmake 3.18
# -DCMAKE_CUDA_ARCHITECTURES:STRING="70" \

# Note: cuda_arch must match the spack.yaml file for the docker image

cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=$SOURCE_DIR/install \
  -DCELERITAS_USE_CUDA:BOOL=ON \
  -DCELERITAS_USE_Geant4:BOOL=ON \
  -DCELERITAS_USE_MPI:BOOL=ON \
  -DCELERITAS_USE_ROOT:BOOL=ON \
  -DCELERITAS_USE_VECGEOM:BOOL=ON \
  -DCELERITAS_BUILD_TESTS:BOOL=ON \
  -DCELERITAS_DEBUG:BOOL=ON \
  -DCMAKE_BUILD_TYPE:STRING="Debug" \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra -pedantic -Werror" \
  -DCMAKE_CUDA_FLAGS="-arch=sm_70 -Werror all-warnings" \
  ..
ninja -v
ctest --output-on-failure
