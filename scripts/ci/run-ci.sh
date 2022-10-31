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

# Set up test arguments
CTEST_TOOL=Test
CTEST_ARGS=""
case ${CMAKE_PRESET} in
  vecgeom-demos )
    CTEST_ARGS="-L app"
    ;;
  valgrind )
    CTEST_TOOL="MemCheck"
    CTEST_ARGS="-LE nomemcheck"
    ;;
  * )
    ;;
esac

# Run tests
cd build
ctest -T ${CTEST_TOOL} ${CTEST_ARGS}\
  -j16 --timeout 180 \
  --no-compress-output --output-on-failure \
  --test-output-size-passed=65536 --test-output-size-failed=1048576 \
# List XML files generated: jenkins will upload these later
find Testing -name '*.xml'
  
if [ "${CMAKE_PRESET}" = "vecgeom-demos" ]; then
  # Test installation
  cd ../scripts/ci
  export LDFLAGS=-Wl,--no-as-needed # for Ubuntu with vecgeom?
  exec ./test-installation.sh
fi
