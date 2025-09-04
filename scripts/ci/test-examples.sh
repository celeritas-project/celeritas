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
  printf "\e[32mBuilding in ${PWD}\e[m\n"
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
if [ -z "${CELER_DISABLE_G4_EXAMPLES}" ]; then
  G4VERSION_STRING="auto"
  if [ -z "${G4VERSION_NUMBER}" ]; then
    # Get the geant4 version 11.2.3, failing if config isn't found
    G4VERSION_STRING=$(geant4-config --version)
    # Replace . with ' ' and convert to MMmp (major/minor/patch)
    G4VERSION_NUMBER=$(echo "${G4VERSION_STRING}" | tr '.' ' ' | xargs printf '%d%01d%01d')
  fi
  echo "Using G4VERSION_NUMBER=\"${G4VERSION_NUMBER}\" (version ${G4VERSION_STRING})"

  # Run small Geant4 examples, ensuring the documentation diff is still valid
  cd "${CELER_SOURCE_DIR}/example/geant4"
  patch -p2 -R < add-celer.diff
  patch -p2 < add-celer.diff
  build_local
  ctest -V --no-tests=error

  if [ "${G4VERSION_NUMBER}" -ge 1070 ]; then
    cd "${CELER_SOURCE_DIR}/example/offload-template"
    build_local
    if [ "${G4VERSION_NUMBER}" -ge 1100 ]; then
      # Run offload-template only on Geant4 v11
      G4FORCENUMBEROFTHREADS=4 G4RUN_MANAGER_TYPE=MT \
        ./run-offload
    else
      # Test that it fails
      echo "*** THE FOLLOWING EXECUTION SHOULD FAIL ***"
      echo "*** (Requires Geant4 11.0 but we have ${G4VERSION_STRING}) ***"
      ./run-offload && (
        echo "Expected run-offload to fail but it PASSED"
        exit 1
      )
      echo "***      EXPECTED FAILURE ABOVE         ***"
      echo "*** *********************************** ***"
    fi
  else
    # RunManagerFactory isn't available
    echo "Skipping offload template: Geant4 version is too old"
  fi
else
  printf "\e[31mSkipping Geant4 tests due to insufficient requirements\e[m\n"
fi
