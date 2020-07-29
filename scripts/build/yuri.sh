#!/bin/sh -e
BUILDSCRIPT_DIR="$(cd "$(dirname $BASH_SOURCE[0])" && pwd)"
SOURCE_DIR="$(cd "${BUILDSCRIPT_DIR}" && git rev-parse --show-toplevel)"

printf "\e[2;37mBuilding from ${SOURCE_DIR}\e[0m\n"
cd $SOURCE_DIR
mkdir build 2>/dev/null || true
cd build

CELERITAS_ENV=/usr/local/spack/var/spack/environments/celeritas/.spack-env/view
export PATH=$CELERITAS_ENV/bin:${PATH}
export CMAKE_PREFIX_PATH=$CELERITAS_ENV:${CMAKE_PREFIX_PATH}

cmake -C ${BUILDSCRIPT_DIR}/yuri.cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX:PATH=$SOURCE_DIR/install \
  ..
ninja -v
