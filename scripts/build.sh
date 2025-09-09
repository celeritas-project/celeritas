#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
#
# Intelligently set up and run CMake using presets and user-defined environment
# scripts that live in `scripts/cmake-presets`. This is intended primarily for
# developers.
#
# 1. Determine the system name
# 2. Source environment file for the system at scripts/env (if available)
# 3. Symbolically link the CMake user settings from scripts/cmake-presets
# 4. Search for ccache
# 5. Forward additional arguments to cmake for configuring
# 6. Build and test
#
#-----------------------------------------------------------------------------#

# Print a colorful message to stderr
log() {
  level=$1
  message=$2

  case "$level" in
    "debug")
      color="2;37;40"
      printf "\033[${color}m%s\033[0m\n" "$message" >&2
      return
      ;;
    "info") color="32;40"; level="INFO" ;;
    "warning") color="33;40"; level="WARNING" ;;
    "error") color="31;40"; level="ERROR" ;;
    *) color="37;40" ;;
  esac

  printf "\033[${color}m%s:\033[0m %s\n" "$level" "$message" >&2
}

# Get the hostname, trying to account for being on a compute node or cluster
fancy_hostname() {
  sys=${LMOD_SYSTEM_NAME}
  if [ -z "${sys}" ]; then
    sys=${HOSTNAME}
    # Trim login/compute from head of string
    case "$sys" in
      login[0-9]*.*|compute[0-9]*.*) sys=${sys#*.} ;;
    esac
    # Trim all but the first component
    sys=${sys%%.*}
  fi
  log debug "Determined system name: ${sys} (override with LMOD_SYSTEM_NAME)"
  echo "${sys}"
}

# Link the presets file
ln_presets() {
  src="scripts/cmake-presets/$1.json"
  dst="CMakeUserPresets.json"

  # Return early if they exist
  if [ -e "${dst}" ]; then
    src=$(readlink ${dst} || echo "<not a symlink>")
    log debug "CMake preset already exists: ${dst} -> ${src}"
    return
  fi

  if [ ! -f "${src}" ]; then
    # Create if
    log info "Creating user presets at ${src} . Please update this file for future configurations."
    cp "scripts/cmake-presets/_dev_.json" "${src}"
    git add "${src}" || error "Could not stage presets"
  fi
  log info "Linking presets to ${dst}"
  ln -s "${src}" "${dst}"
}

#
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
  if CCACHE_PROGRAM="$(which ccache)" >/dev/null 2>&1; then
    export CCACHE_PROGRAM="$(which ccache)"
    log info "Using ccache: ${CCACHE_PROGRAM}"
    check_ccache_usage
  fi
  export CCACHE_PROGRAM=${CCACHE_PROGRAM}
}

#-----------------------------------------------------------------------------#

# Run everything from the parent directory of this script (i.e. the Celeritas
# source dir)
cd "$(dirname $0)"/..

# Determine system name
SYSTEM_NAME=$(fancy_hostname)

# Load environment paths
_env_script="scripts/env/${SYSTEM_NAME}.sh"
if [ -f "${_env_script}" ]; then
  log info "Sourcing environment script at ${_env_script}"
  . "${_env_script}"
else
  log debug "No environment script exists at ${_env_script}"
fi

# Link preset file
ln_presets "${SYSTEM_NAME}"

# Search for and probe ccache
setup_ccache

# Check arguments and give presets if missing
if [ $# -eq 0 ]; then
  echo "Usage: $0 PRESET [config_args...]" >&2
  if hash cmake 2>/dev/null ; then
    cmake --list-presets >&2 \
      || log error "CMake may be too old or JSON file may be broken"
  else
    log error "cmake unavailable: cannot call --list-presets"
  fi
  exit 2
fi
CMAKE_PRESET=$1
shift

# Configure, build, and test
log info "Configuring with verbosity"
cmake --preset=${CMAKE_PRESET} --log-level=VERBOSE "$@"
log info "Building"
cmake --build --preset=${CMAKE_PRESET}
log info "Testing"
ctest --preset=${CMAKE_PRESET}

log info "Celeritas was successfully built and tested for development!"
