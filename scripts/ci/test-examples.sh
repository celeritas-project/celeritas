#!/bin/sh -e
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$(dirname $0)"/../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
fi
export CMAKE_PREFIX_PATH=${CELER_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

test -d "${CELER_INSTALL_DIR}" || (
  echo "CELER_INSTALL_DIR=${CELER_INSTALL_DIR} is not a directory"
  exit 1
)
test -n "${CMAKE_PRESET}" || (
  echo "CMAKE_PRESET is undefined"
  exit 1
)

build_local() {
  git clean -fxd .
  EXAMPLE_INSTALL=${PWD}/install
  printf "\e[32mTesting in ${PWD}\e[m\n"
  mkdir build
  cd build
  cmake -G Ninja \
    -D CMAKE_INSTALL_PREFIX=${EXAMPLE_INSTALL} \
    ..
  ninja
}

export CELER_LOG=debug CELER_LOG_LOCAL=debug

# Run minimal example
cd "${CELER_SOURCE_DIR}/example/minimal"
build_local
./minimal

# Run Geant4 app examples
if [ -z "${CELER_DISABLE_ACCEL_EXAMPLES}" ]; then
  if [ -z "${G4VERSION_NUMBER}" ]; then
    # Get the geant4 version 11.2.3, replace `.` with ` `, concat as 110203
    _vers=$(geant4-config --version)
    G4VERSION_NUMBER=$(printf '%d%02d%02d' ${_vers//./ })
    echo "Set G4VERSION_NUMBER=${G4VERSION_NUMBER}"
  fi

  # Run small accel examples
  cd "${CELER_SOURCE_DIR}/example/accel"
  build_local
  ctest -V --no-tests=error

  if [ "${G4VERSION_NUMBER}" -ge 1100 ]; then
    # Run offload-template only on Geant4 v11
    cd "${CELER_SOURCE_DIR}/example/offload-template"
    build_local
    ./run-offload
  fi
else
  printf "\e[31mSkipping 'accel' tests due to insufficient requirements\e[m\n"
fi
