#!/bin/sh -e

CMAKE_PRESET=$1
shift

_ctest_args="-j16 --timeout 180 --no-compress-output --test-output-size-passed=65536 --test-output-size-failed=1048576"
if [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  _ctest_args="-L 'app' ${_ctest_args}"
fi

set -x

cd "$(dirname $0)"/../..
ln -fs scripts/cmake-presets/ci.json CMakeUserPresets.json

# Fetch tags for version provenance
git fetch --tags
# Configure
cmake --preset=${CMAKE_PRESET}
# Build and (optionally) install
cmake --build --preset=${CMAKE_PRESET}

# Require regression-like tests to be enabled and pass
export CELER_TEST_STRICT=1

cd build
# Test
ctest -T Test ${_ctest_args}
if [ "${CMAKE_PRESET}" = "valgrind" ]; then
  # Run Valgrind, but skip apps that are launched through python drivers
  ctest -T MemCheck -LE nomemcheck --output-on-failure \
    ${_ctest_args}
elif [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  cd ci
  exec test-installation.sh
fi
