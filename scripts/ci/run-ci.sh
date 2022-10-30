#!/bin/sh -e

CMAKE_PRESET=$1
shift

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

if ! [ "${CMAKE_PRESET}" = "valgrind" ]; then
  # Regular testing
  ctest --preset=${CMAKE_PRESET} --no-compress-output
else
  # Run Valgrind, but skip apps that are launched through python drivers
  cd build
  ctest -T MemCheck -LE nomemcheck --output-on-failure --timeout=180
fi

if [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  # Test installation
  cd scripts/ci
  exec test-installation.sh
fi
