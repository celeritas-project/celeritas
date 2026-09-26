#!/bin/sh -e
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$(dirname $0)"/../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
  echo "CELER_INSTALL_DIR is undefined: using ${CELER_INSTALL_DIR}"
fi
if [ -z "${CELER_CMAKE_PRESET}" ]; then
  CELER_CMAKE_PRESET="base"
  echo "CELER_CMAKE_PRESET is undefined: using ${CELER_CMAKE_PRESET}"
fi
if [ -z "${G4VERSION_NUMBER}" ]; then
  if ! _g4config_exe=$(command -v geant4-config) ; then
    echo "Could not find Geant4 version: define G4VERSION_NUMBER=0 to disable G4 tests"
    exit 1
  fi
  # Replace . with ' ' and convert to MMmp (major/minor/patch)
  G4VERSION_NUMBER=$(${_g4config_exe} --version | tr '.' ' ' | xargs printf '%d%01d%01d')
  echo "Set G4VERSION_NUMBER=${G4VERSION_NUMBER} from ${_g4config_exe}"
fi
export CMAKE_PREFIX_PATH=${CELER_INSTALL_DIR}:${CMAKE_PREFIX_PATH}

test -d "${CELER_INSTALL_DIR}" || {
  echo "CELER_INSTALL_DIR=${CELER_INSTALL_DIR} is not a directory"
  exit 1
}

build_local() {
  git clean -fxd .
  EXAMPLE_INSTALL=${PWD}/install
  printf "\033[32mBuilding in ${PWD}\033[m\n"
  mkdir build
  cd build
  cmake -G Ninja --log-level=verbose \
    -D CMAKE_INSTALL_PREFIX=${EXAMPLE_INSTALL} \
    ..
  ninja
}
export CELER_LOG=debug CELER_LOG_LOCAL=debug

echo "::group::Build and run minimal example"
cd "${CELER_SOURCE_DIR}/example/minimal"
build_local
./minimal
echo "::endgroup::"

# Run Geant4 app examples unless DISABLE is set to a non-empty, non-zero value
if [ "${G4VERSION_NUMBER}" -eq 0 ]; then
  printf "\033[31mSkipping Geant4 test: G4VERSION_NUMBER=0\033[m\n"
  exit 0
fi

### WHEN USING GEANT4 ###

echo "Using G4VERSION_NUMBER=\"${G4VERSION_NUMBER}\""

# Run small Geant4 examples, ensuring the documentation diff is still valid
cd "${CELER_SOURCE_DIR}/example/geant4"
patch -p2 -R < add-celer.diff
patch -p2 < add-celer.diff
echo "::group::Build and run basic G4 examples"
build_local
ctest -V --no-tests=error
echo "::endgroup::"

if [ "${G4VERSION_NUMBER}" -lt 1070 ]; then
  # RunManagerFactory isn't available
  echo "Skipping offload template tests: Geant4 version is too old"
  exit 0
fi

### WHEN USING GEANT4 10.7+ ###

echo "::group::Build basic offload template"
cd "${CELER_SOURCE_DIR}/example/offload-template"
build_local
echo "::endgroup::"
if [ "${G4VERSION_NUMBER}" -lt 1100 ]; then
  # Test that it fails
  echo "*** THE FOLLOWING EXECUTION SHOULD FAIL ***"
  echo "*** (Requires Geant4 11.0 but we have ${G4VERSION_NUMBER}) ***"
  if ./run-offload > offload-should-fail.txt 2>&1 ; then
    cat offload-should-fail.txt
    echo "Expected run-offload to fail but it PASSED"
    exit 1
  fi
  echo "::group::Offload output"
  cat offload-should-fail.txt
  echo "::endgroup::"
  echo "Run-offload failed as expected"
  exit 0
fi

### WHEN USING GEANT4 11.0+ ###

# Run offload-template only on Geant4 v11
echo "::group::Run offload template"
G4FORCENUMBEROFTHREADS=4 G4RUN_MANAGER_TYPE=MT \
  ./run-offload
echo "::endgroup::"
