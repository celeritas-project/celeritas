#!/bin/sh -e

BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
source "${BUILDSCRIPT_DIR}/docker-setup.sh"

set -x

git -C ${SOURCE_DIR} fetch -f --tags

# Note: cuda_arch must match the spack.yaml file for the docker image, which
# must match the hardware being used.
cmake -G Ninja \
  -DCELERITAS_BUILD_DEMOS:BOOL=ON \
  -DCELERITAS_BUILD_TESTS:BOOL=ON \
  -DCELERITAS_USE_CUDA:BOOL=ON \
  -DCELERITAS_USE_Geant4:BOOL=ON \
  -DCELERITAS_USE_HepMC3:BOOL=ON \
  -DCELERITAS_USE_MPI:BOOL=ON \
  -DCELERITAS_USE_ROOT:BOOL=ON \
  -DCELERITAS_USE_VecGeom:BOOL=ON \
  -DCELERITAS_DEBUG:BOOL=ON \
  -DCMAKE_BUILD_TYPE:STRING="Debug" \
  -DCMAKE_CUDA_ARCHITECTURES:STRING="70" \
  -DCMAKE_EXE_LINKER_FLAGS:STRING="-Wl,--no-as-needed" \
  -DCMAKE_SHARED_LINKER_FLAGS:STRING="-Wl,--no-as-needed" \
  -DMPIEXEC_PREFLAGS:STRING="--allow-run-as-root" \
  -DMPI_CXX_LINK_FLAGS:STRING="-pthread" \
  ${SOURCE_DIR}
cmake --build . -j${PARALLEL_LEVEL} -- -k0
ctest $CTEST_ARGS
