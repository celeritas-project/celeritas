#!/bin/sh -ex
# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$(dirname $0)"/../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
fi
export CMAKE_PREFIX_PATH=${CELER_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

test -n "${CMAKE_PRESET}" || (echo "CMAKE_PRESET is undefined"; exit 1)

build_local() {
  git clean -fxd .
  EXAMPLE_INSTALL=${PWD}/install
  mkdir build
  cd build
  cmake -G Ninja \
    -D CMAKE_INSTALL_PREFIX=${EXAMPLE_INSTALL} \
    ..
  ninja
}

cd "${CELER_SOURCE_DIR}/example/minimal"
build_local
./minimal

if [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  # The 'accel' example requires Geant4 and VecGeom
  cd "${CELER_SOURCE_DIR}/example/accel"
  build_local
  ctest -V --no-tests=error
fi
