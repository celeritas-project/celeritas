#!/bin/sh -e

_ctest_args="-j16 --no-compress-output --test-output-size-passed=65536 --test-output-size-failed=1048576"

set -x

cd "$(dirname $0)"/../../..
ln -fs scripts/cmake-presets/ci.json CMakeUserPresets.json

CMAKE_PRESET=$1
shift

# Configure
cmake --preset=${CMAKE_PRESET}
# Build
cmake --build --preset=${CMAKE_PRESET}

cd build
# Test
ctest -T Test --timeout 180 ${_ctest_args}
# Valgrind
if [ "${CMAKE_PRESET}" = "valgrind" ]; then
  ctest -T MemCheck -E 'demo-' --timeout 180 ${_ctest_args}
fi
