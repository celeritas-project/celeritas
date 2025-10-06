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

  # Create if preset file doesn't exist
  if [ ! -f "${src}" ]; then
    log info "Creating user presets at ${src} . Please update this file for future configurations."
    cp "scripts/cmake-presets/_dev_.json" "${src}"
    git add "${src}" || error "Could not stage presets"
  fi
  log info "Linking presets to ${dst}"
  ln -s "${src}" "${dst}"
}

# Check if ccache is full and warn user
check_ccache_usage() {
  cache_stats=$(ccache --print-stats)
  current_kb=$(echo "$cache_stats" | grep "^cache_size_kibibyte" | cut -f2)
  max_kb=$(echo "$cache_stats" | grep "^max_cache_size_kibibyte" | cut -f2)

  if [ -n "$current_kb" ] && [ -n "$max_kb" ] && [ "$max_kb" -gt 0 ] 2>/dev/null; then
    usage_percent=$((current_kb * 100 / max_kb))

    if [ "$usage_percent" -gt 90 ] 2>/dev/null; then
      # Convert to human-readable sizes (GiB)
      current_gb=$((current_kb / 1024 / 1024))
      max_gb=$((max_kb / 1024 / 1024))
      log warning "ccache is ${usage_percent}% full (${current_gb}/${max_gb} GiB)"
      log info "Consider increasing cache size with 'ccache -M <size>G' or clearing with 'ccache -C'"
    fi
  fi
}

# Auto-detect and configure ccache if available
setup_ccache() {
  if CCACHE_PROGRAM="$(which ccache 2>/dev/null)"; then
    log info "Using ccache: ${CCACHE_PROGRAM}"
    check_ccache_usage
  fi
  export CCACHE_PROGRAM=${CCACHE_PROGRAM}
}

# Check if pre-commit hook is installed and install if missing
install_precommit_if_git() {
  git_dir=$(git rev-parse --git-dir 2>/dev/null)
  if [ $? -ne 0 ]; then
    log debug "Not in a git repository, skipping pre-commit check"
    return 1
  fi

  if [ ! -f "${git_dir}/hooks/pre-commit" ]; then
    log info "Pre-commit hook not found, installing commit hooks"
    ./scripts/dev/install-commit-hooks.sh
  fi
  return 0
}

#-----------------------------------------------------------------------------#

# Run everything from the parent directory of this script (i.e. the Celeritas
# source dir)
cd "$(dirname $0)"/..

# Determine system name
SYSTEM_NAME=$(fancy_hostname)

# Check whether cmake changes from environment
OLD_CMAKE=$(which cmake 2>/dev/null || echo "")
OLD_PRE_COMMIT=$(which pre-commit 2>/dev/null)

# Load environment paths
_env_script="scripts/env/${SYSTEM_NAME}.sh"
if [ -f "${_env_script}" ]; then
  log info "Sourcing environment script at ${_env_script}"
  . "${_env_script}"
else
  log debug "No environment script exists at ${_env_script}"
fi

NEW_CMAKE=$(which cmake 2>/dev/null || echo "cmake unavailable")

# Link preset file
ln_presets "${SYSTEM_NAME}"

# Search for and probe ccache
setup_ccache

# Check arguments and give presets if missing
if [ $# -eq 0 ]; then
  echo "Usage: $0 PRESET [config_args...]" >&2
  if hash cmake 2>/dev/null ; then
    if cmake --list-presets >&2 ; then
      log info "Try using the '${SYSTEM_NAME}' or 'dev' presets ('base'), or 'default' if not"
    else
      log error "CMake may be too old or JSON file may be broken"
    fi
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
if cmake --build --preset=${CMAKE_PRESET}; then
  log info "Testing"
  if ctest --preset=${CMAKE_PRESET} --timeout 15; then
    log info "Celeritas was successfully built and tested for development!"
  else
    log warning "Celeritas built but some tests failed"
    log info "Ask the Celeritas team whether the failures indicate an actual error"
  fi

  install_precommit_if_git
  if [ "${NEW_CMAKE}" != "${OLD_CMAKE}" ]; then
    log warning "Local environment script uses a different CMake than your \$PATH:"
    log info "Recommend adding '. ${PWD}/${_env_script}' to your shell rc"
  fi
else
  log error "build failed: check configuration and build errors above"
fi
