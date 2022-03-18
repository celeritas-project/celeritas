#!/bin/sh -e

cd "$(dirname $0)"/..

# Link user presets for this system if they don't exist
if [ ! -e "CMakeUserPresets.json" ]; then
  if [ -z "${LMOD_SYSTEM_NAME}" ]; then
    LMOD_SYSTEM_NAME=${HOSTNAME%%.*}
  fi
  _USER_PRESETS="scripts/cmake-presets/${LMOD_SYSTEM_NAME}.json"
  if [ -f "${_USER_PRESETS}" ]; then
    ln -s "${_USER_PRESETS}" "CMakeUserPresets.json"
  fi
fi

# Check arguments and give presets
if [ $# -eq 0 ]; then
  echo "Usage: $0 PRESET [config_args...]" >&2
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
cmake --preset=${CMAKE_PRESET} "$@"
# Build
cmake --build --preset=${CMAKE_PRESET}
# Test
ctest --preset=${CMAKE_PRESET} 
