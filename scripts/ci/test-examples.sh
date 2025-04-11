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
  cmake -G Ninja --log-level=verbose \
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
    # Get the geant4 version 11.2.3, failing if config isn't found
    _vers=$(geant4-config --version)
    # Replace . with ' ' and convert to MMmp (major/minor/patch)
    G4VERSION_NUMBER=$(echo "${_vers}" | tr '.' ' ' | xargs printf '%d%01d%01d')
    echo "Set G4VERSION_NUMBER=\"${G4VERSION_NUMBER}\""
  fi

  # Run small accel examples, ensuring the documentation diff is still valid
  cd "${CELER_SOURCE_DIR}/example/accel"
  patch -p2 -R < add-celer.diff
  patch -p2 < add-celer.diff
  build_local
  ctest -V --no-tests=error

  if [ "${G4VERSION_NUMBER}" -ge 1100 ]; then
    # Run offload-template only on Geant4 v11
    cd "${CELER_SOURCE_DIR}/example/offload-template"
    build_local
    G4FORCENUMBEROFTHREADS=4 G4RUN_MANAGER_TYPE=MT \
      ./run-offload
  fi
else
  printf "\e[31mSkipping 'accel' tests due to insufficient requirements\e[m\n"
fi
