#!/bin/sh -e

OS=$1
CMAKE_PRESET=$2

set -x
test -n "${OS}"
test -n "${CMAKE_PRESET}"

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR="$(dirname $0)"/../..
fi
cd "${CELER_SOURCE_DIR}"
CELER_SOURCE_DIR=$(pwd)

# Source environment script if necessary
_ENV_SCRIPT="scripts/env/ci-${OS}.sh"
if [ -f "${_ENV_SCRIPT}" ]; then
  . "${_ENV_SCRIPT}"
fi

# Fetch tags for version provenance
git fetch -f --tags

# Clean older builds from jenkins *BEFORE* setting up presets
git clean -fxd
# Stale tmp files?
rm -rf /tmp/ompi.*

# Link preset script
ln -fs scripts/cmake-presets/ci-${OS}.json CMakeUserPresets.json
# Configure
cmake --preset=${CMAKE_PRESET}
# Build and (optionally) install
cmake --build --preset=${CMAKE_PRESET}

# Require regression-like tests to be enabled and pass
export CELER_TEST_STRICT=1

# Run tests
cd build
ctest -T Test \
  -j16 --timeout 180 \
  --no-compress-output --output-on-failure \
  --test-output-size-passed=65536 --test-output-size-failed=1048576 \
# List XML files generated: jenkins will upload these later
find Testing -name '*.xml'

# Install and test that the executables are there
cmake --install .
cd ..
test -x "${CELER_SOURCE_DIR}/install/bin/celer-sim"
test -x "${CELER_SOURCE_DIR}/install/bin/celer-g4"
"${CELER_SOURCE_DIR}/install/bin/celer-sim" --version


# Test examples against installed celeritas
export CMAKE_PRESET
export CELER_SOURCE_DIR
case "${CMAKE_PRESET}" in
  *-vecgeom*)
    # VecGeom is in use: ubuntu flags are too strict for it
    export LDFLAGS=-Wl,--no-as-needed
    ;;
esac

# Test examples against installed celeritas
exec ${CELER_SOURCE_DIR}/scripts/ci/test-examples.sh
