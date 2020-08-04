#!/bin/sh -ex
###############################################################################
# File  : scripts/travis/configure.sh
###############################################################################

cd ${BUILD_DIR} && cmake  \
  -D CELERITAS_USE_CUDA=OFF \
  -D CELERITAS_USE_MPI=OFF \
  -D CELERITAS_USE_ROOT=OFF \
  -D CELERITAS_USE_VECGEOM=OFF \
  -D CMAKE_BUILD_TYPE="RelWithDebInfo" \
  -D CMAKE_CXX_FLAGS="-Wall -Wextra -Werror -pedantic" \
  -D CMAKE_INSTALL_PREFIX="${CELERITAS_DIR}" \
  -D MEMORYCHECK_COMMAND_OPTIONS="--error-exitcode=1 --leak-check=full" \
  ${SOURCE_DIR}

###############################################################################
# end of scripts/travis/configure.sh
###############################################################################
