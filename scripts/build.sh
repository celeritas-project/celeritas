#!/bin/sh
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

set -e

# Print a colorful message to stderr
celerlog() {
  level=$1
  message=$2

  case "${level}" in
    debug)
      color="2;37;40"
      printf "\033[%sm%s\033[0m\n" "${color}" "${message}" >&2
      return
      ;;
    info) color="32;40"; level="INFO" ;;
    warning) color="33;40"; level="WARNING" ;;
    error) color="31;40"; level="ERROR" ;;
    *) color="37;40" ;;
  esac

  printf "\033[%sm%s:\033[0m %s\n" "${color}" "${level}" "${message}" >&2
}

log() {
  celerlog "$@"
}

# Get the hostname, trying to account for being on a compute node or cluster
fancy_hostname() {
  sys=${LMOD_SYSTEM_NAME}
  if [ -z "${sys}" ]; then
    sys="$(uname -n)"
    # Trim login/compute from head of string
    case "${sys}" in
      login[0-9]*.*|compute[0-9]*.*) sys=${sys#*.} ;;
    esac
    # Trim all but the first component
    sys=${sys%%.*}
  fi
  log debug "Determined system name: ${sys} (override with LMOD_SYSTEM_NAME)"
  printf '%s\n' "${sys}"
}

# Load environment (make sure this is not executed as a subshell)
load_system_env() {
  ENV_SCRIPT="${CELER_SOURCE_DIR}/scripts/env/$1.sh"
  if [ -f "${ENV_SCRIPT}" ]; then
    log info "Sourcing environment script at ${ENV_SCRIPT}"
    set +e
    if ! . "$ENV_SCRIPT"; then
      log error "Environment setup for $1 failed"
      exit 1
    fi
    set -e
  else
    log debug "No environment script exists at ${ENV_SCRIPT}"
  fi
}

# Link the presets file
ln_presets() {
  src="scripts/cmake-presets/$1.json"
  dst="CMakeUserPresets.json"

  # Return early if it exists
  if [ -L "${dst}" ]; then
    src="$(readlink "${dst}" 2>/dev/null || printf "<unknown>")"
    log debug "CMake preset already exists: ${dst} -> ${src}"
    return
  elif [ -e "${dst}" ]; then
    log debug "CMake preset already exists: ${dst}"
    return
  fi

  # Create preset if it doesn't exist
  if [ ! -f "${src}" ]; then
    log info "Creating user presets at ${src} . Please update this file for future configurations."
    cp "scripts/cmake-presets/_dev_.json" "${src}"
    git add "${src}" || log error "Could not stage presets"
  fi
  log info "Linking presets to ${dst}"
  ln -s "${src}" "${dst}"
}

# Check if ccache is full and warn user
check_ccache_usage() {
  cache_stats=$(ccache --print-stats 2>/dev/null || printf '')
  current_kb=$(printf '%s\n' "${cache_stats}" | grep "^cache_size_kibibyte" | cut -f2)
  max_kb=$(printf '%s\n' "${cache_stats}" | grep "^max_cache_size_kibibyte" | cut -f2)

  # Ensure numeric
  case ${current_kb} in (""|*[!0-9]*) current_kb= ;; esac
  case ${max_kb} in (""|*[!0-9]*) max_kb= ;; esac

  if [ -n "${current_kb}" ] && [ -n "${max_kb}" ] && [ "${max_kb}" -gt 0 ]; then
    usage_percent=$((current_kb * 100 / max_kb))
    if [ "${usage_percent}" -gt 90 ]; then
      current_gb=$((current_kb / 1024 / 1024))
      max_gb=$((max_kb / 1024 / 1024))
      log warning "ccache is ${usage_percent}% full (${current_gb}/${max_gb} GiB)"
      log info "Consider increasing cache size with 'ccache -M <size>G' or clearing with 'ccache -C'"
    fi
  fi
}

# Auto-detect and configure ccache if available
setup_ccache() {
  if CCACHE_PROGRAM="$(command -v ccache 2>/dev/null)"; then
    log info "Using ccache: ${CCACHE_PROGRAM}"
    check_ccache_usage
    [ -n "${CCACHE_PROGRAM}" ] && export CCACHE_PROGRAM
  fi
}

# Check if pre-commit hook is installed and install if missing
install_precommit_if_git() {
  if git_dir=$(git rev-parse --git-dir 2>/dev/null); then
    :
  else
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

# Determine the source directory from the build script
export CELER_SOURCE_DIR="$(cd "$(dirname "$0")"/.. && pwd)"

# Run everything from the Celeritas work tree
cd ${CELER_SOURCE_DIR}

# Determine system name, failing on an empty string
SYSTEM_NAME=$(fancy_hostname)
if [ -z "${SYSTEM_NAME}" ]; then
  log warning "Could not determine SYSTEM_NAME from LMOD_SYSTEM_NAME or HOSTNAME"
  log error "Empty SYSTEM_NAME, needed to load environment and presets"
  exit 1
fi

# Check whether cmake/pre-commit change from environment
OLD_CMAKE="$(command -v cmake 2>/dev/null || printf '')"
OLD_PRE_COMMIT="$(command -v pre-commit 2>/dev/null || printf '')"

# Load environment paths using the system name
load_system_env "${SYSTEM_NAME}"

NEW_CMAKE="$(command -v cmake 2>/dev/null || printf 'cmake unavailable')"
NEW_PRE_COMMIT="$(command -v pre-commit 2>/dev/null || printf '')"

# Link preset file
ln_presets "${SYSTEM_NAME}"

# Search for and probe ccache
setup_ccache

# Check arguments and give presets if missing
if [ $# -eq 0 ]; then
  printf '%s\n' "Usage: $0 PRESET [config_args...]" >&2
  if command -v cmake >/dev/null 2>&1; then
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
cmake --preset="${CMAKE_PRESET}" --log-level=VERBOSE "$@"
log info "Building"
if cmake --build --preset="${CMAKE_PRESET}"; then
  log info "Testing"
  if ctest --preset="${CMAKE_PRESET}" --timeout 15; then
    log info "Celeritas was successfully built and tested for development!"
  else
    log warning "Celeritas built but some tests failed"
    log info "Ask the Celeritas team whether the failures indicate an actual error"
  fi

  install_precommit_if_git
  if [ "${NEW_PRE_COMMIT}" != "${OLD_PRE_COMMIT}" ]; then
    log warning "Local environment script uses a different pre-commit than your \$PATH:"
    log info "Recommend adding '. ${CELER_SOURCE_DIR}/${ENV_SCRIPT}' to your shell rc"
  fi

  if [ "${NEW_CMAKE}" != "${OLD_CMAKE}" ]; then
    log warning "Local environment script uses a different CMake than your \$PATH:"
    log info "Recommend adding '. ${CELER_SOURCE_DIR}/${ENV_SCRIPT}' to your shell rc"
  fi
else
  log error "build failed: check configuration and build errors above"
fi
