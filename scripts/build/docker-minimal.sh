#!/bin/sh -e

BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
source "${BUILDSCRIPT_DIR}/docker-setup.sh"

set -x

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
  -DCMAKE_BUILD_TYPE:STRING="RelWithDebInfo" \
  -DMEMORYCHECK_COMMAND_OPTIONS="--error-exitcode=1 --leak-check=full" \
  ${SOURCE_DIR}
ninja -v -k0
ctest $CTEST_ARGS

ctest -E demo- -D ExperimentalMemCheck --output-on-failure
