#!/bin/sh -e

cd "$(dirname $0)"/..

SYSTEM_NAME=${LMOD_SYSTEM_NAME}
if [ -z "${SYSTEM_NAME}" ]; then
  SYSTEM_NAME=${HOSTNAME%%.*}
  SYSTEM_NAME=${SYSTEM_NAME%%login*}
fi

# Link user presets for this system if they don't exist
if [ ! -e "CMakeUserPresets.json" ]; then
  _USER_PRESETS="scripts/cmake-presets/${SYSTEM_NAME}.json"
  if [ -f "${_USER_PRESETS}" ]; then
    ln -s "${_USER_PRESETS}" "CMakeUserPresets.json"
  fi
fi

# Source environment script if necessary
_ENV_SCRIPT="scripts/env/${SYSTEM_NAME}.sh"
if [ -f "${_ENV_SCRIPT}" ]; then
  echo "Sourcing environment script at ${_ENV_SCRIPT}" >&2
  . "${_ENV_SCRIPT}"
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

cmake --preset=${CMAKE_PRESET} "$@"
cmake --build --preset=${CMAKE_PRESET}
ctest --preset=${CMAKE_PRESET}
