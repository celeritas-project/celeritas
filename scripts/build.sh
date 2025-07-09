#!/bin/sh -e

log() {
    level=$1
    message=$2

    case "$level" in
        "info") color="32;40"; level="INFO" ;;
        "warning") color="33;40"; level="WARNING" ;;
        "error") color="31;40"; level="ERROR" ;;
        *) color="2;37;40" ;;
    esac

    printf "\033[${color}m%s:\033[0m %s\n" "$level" "$message" >&2
}

check_ccache_usage() {
    # Check if ccache is full and warn user
    cache_stats=$(ccache --print-stats)
    current_kb=$(echo "$cache_stats" | grep "^cache_size_kibibyte" | cut -f2)
    max_kb=$(echo "$cache_stats" | grep "^max_cache_size_kibibyte" | cut -f2)

    if [ -n "$current_kb" ] && [ -n "$max_kb" ] && [ "$max_kb" -gt 0 ] 2>/dev/null; then
        usage_percent=$((current_kb * 100 / max_kb))

        if [ "$usage_percent" -gt 90 ] 2>/dev/null; then
            # Convert to human-readable sizes (GiB)
            current_gb=$((current_kb / 1024 / 1024))
            max_gb=$((max_kb / 1024 / 1024))
            log warning "ccache is ${usage_percent}% full (${current_gb}/${max_gb} GiB). Consider increasing cache size with 'ccache -M <size>G'"
        fi
    fi
}

setup_ccache() {
    # Auto-detect and configure ccache if available
    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_PROGRAM="$(which ccache)"
        log info "Using ccache"
        check_ccache_usage
    else
        export CCACHE_PROGRAM=""
    fi
}

cd "$(dirname $0)"/..

SYSTEM_NAME=${LMOD_SYSTEM_NAME}
if [ -z "${SYSTEM_NAME}" ]; then
  SYSTEM_NAME=${HOSTNAME%%.*}
  SYSTEM_NAME=${SYSTEM_NAME%%login*}
fi

# Link user presets for this system if they don't exist
if [ ! -e "CMakeUserPresets.json" ]; then
  _USER_PRESETS="scripts/cmake-presets/${SYSTEM_NAME}.json"
  if [ ! -f "${_USER_PRESETS}" ]; then
    log info "Creating user presets at ${_USER_PRESETS} . Please update this file for future configurations."
    cp "scripts/cmake-presets/_dev_.json" "${_USER_PRESETS}"
    git add "${_USER_PRESETS}" || error "Could not add presets"
  fi
  log info "Linking presets to ./CMakeUserPresets.json"
  ln -s "${_USER_PRESETS}" "CMakeUserPresets.json"
fi

# Source environment script if necessary
_ENV_SCRIPT="scripts/env/${SYSTEM_NAME}.sh"
if [ -f "${_ENV_SCRIPT}" ]; then
  log info "Sourcing environment script at ${_ENV_SCRIPT}" >&2
  . "${_ENV_SCRIPT}"
fi

# Setup ccache
setup_ccache

# Check arguments and give presets
if [ $# -eq 0 ]; then
  echo "Usage: $0 PRESET [config_args...]" >&2
  if hash cmake 2>/dev/null ; then
    cmake --list-presets >&2 \
      || log error "CMake may be too old or JSON file may be broken"
  else
    log error "cmake unavailable: cannot call --list-presets" >&2
  fi
  exit 2
fi
CMAKE_PRESET=$1
shift

log info "Configuring with verbosity"
cmake --preset=${CMAKE_PRESET} --log-level=VERBOSE "$@"
log info "Building"
cmake --build --preset=${CMAKE_PRESET}
log info "Testing"
ctest --preset=${CMAKE_PRESET}
