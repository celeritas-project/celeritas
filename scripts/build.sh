#!/bin/sh -e

if [ -z "${SOURCE_DIR}" ]; then
  cd "$(cd "$(dirname $0)" && git rev-parse --show-toplevel)"
else
  cd "${SOURCE_DIR}"
fi


# Link user presets for this system if they don't exist
if [ ! -e "CMakeUserPresets.json" ]; then
  _USER_PRESETS="scripts/cmake-presets/${HOSTNAME%%.*}.json"
  if [ -f "${_USER_PRESETS}" ]; then
    ln -s "${_USER_PRESETS}" "CMakeUserPresets.json"
  fi
fi

# Check arguments and give presets
if [ $# -ne 1 ]; then
  echo "Usage: $0 PRESET" >&2
  if hash cmake 2>/dev/null ; then
    cmake --list-presets >&2
  else
    echo "cmake unavailable: cannot call --list-presets" >&2
  fi
  exit 2
fi
CMAKE_PRESET=$1
shift

set -x

# Configure
cmake --preset=${CMAKE_PRESET}
# Build
cmake --build --preset=${CMAKE_PRESET}
# Test
ctest --preset=${CMAKE_PRESET}
