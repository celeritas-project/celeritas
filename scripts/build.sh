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

# Determine the environment script for the given hostname
get_system_env() {
  ENV_SCRIPT="${CELER_SOURCE_DIR}/scripts/env/$1.sh"
  if [ ! -f "${ENV_SCRIPT}" ]; then
    log debug "No environment script exists at ${ENV_SCRIPT}"
  else
    log info "Found environment script at ${ENV_SCRIPT}"
    printf '%s\n' "${ENV_SCRIPT}"
  fi
}

# Source a script safely (make sure this is not executed as a subshell)
source_script() {
  set +e
  if ! . "$1"; then
    log error "Failed to source '$1'"
    exit 1
  fi
  _celer_build=0
  set -e
}

# Allow downstream scripts (e.g., excl) to load additional source files
load_system_env() {
  env_script_="$(get_system_env "${1}")"
  if [ -z "${env_script_}" ]; then
    log error "Failed to find environment script for '$1'"
    return 1
  else
    source_script "${env_script_}"
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

# Find a command in PATH, suppressing error messages
find_command() {
    command -v "$1" 2>/dev/null || printf ""
}

# Auto-detect and configure ccache if available
setup_ccache() {
  CCACHE_PROGRAM="$(find_command ccache)"
  if [ -n "${CCACHE_PROGRAM}" ]; then
    log info "Using ccache: ${CCACHE_PROGRAM}"
    check_ccache_usage
    export CCACHE_PROGRAM
  fi
}

# Check if pre-commit hook is installed and install if missing
install_precommit_if_git() {
  if ! git_hook_path="$(git rev-parse --git-path hooks/pre-commit 2>/dev/null)"; then
    log debug "Not in a git repository, skipping pre-commit check"
    return 1
  fi
  log debug "Checking for pre-commit hooks at '${git_hook_path}'"

  if [ ! -f "${git_hook_path}" ]; then
    log info "Pre-commit hook not found, installing commit hooks"
    ./scripts/dev/install-commit-hooks.sh
  fi
  return 0
}

# Determine the RC file for the current shell
get_shell_rc_file() {
  shell_name=$(basename "${SHELL}")
  case "${shell_name}" in
    bash) printf '%s\n' "${HOME}/.bashrc" ;;
    zsh) printf '%s\n' "${HOME}/.zshrc" ;;
    ksh) printf '%s\n' "${HOME}/.kshrc" ;;
    *) log warning "Unknown shell: ${shell_name}"; printf '%s\n' "${HOME}/.bashrc" ;;
  esac
}

sentinel_marker_='celeritas-build-env'

# Check if sentinel marker exists in rc file
has_sentinel() {
  rc_file=$1
  if [ ! -f "${rc_file}" ]; then
    return 1
  fi
  grep -q "${sentinel_marker_}" "${rc_file}" 2>/dev/null
}

# Install environment sourcing to shell rc file
install_shell_env() {
  rc_file=$1

  if has_sentinel "${rc_file}"; then
    log debug "Skipping modification of ${rc_file}: sentinel exists"
    return 1
  fi

  log info "Installing environment sourcing to ${rc_file}"
  cat >> "${rc_file}" <<EOF
# >>> ${sentinel_marker_} >>>
load_system_env() {
  CELER_SOURCE_DIR="${CELER_SOURCE_DIR}"
  if [ -d "\${CELER_SOURCE_DIR}" ]; then
    . "\${CELER_SOURCE_DIR}/scripts/env/\$1.sh"
  fi
}
load_system_env "${SYSTEM_NAME}"
# <<< ${sentinel_marker_} <<<
EOF
}

#-----------------------------------------------------------------------------#

# Determine the source directory from the build script
export CELER_SOURCE_DIR="$(cd "$(dirname "$0")"/.. && pwd)"

# Run everything from the Celeritas work tree
cd ${CELER_SOURCE_DIR}

# Determine system name, failing on an empty string
SYSTEM_NAME="$(fancy_hostname)"
if [ -z "${SYSTEM_NAME}" ]; then
  log warning "Could not determine SYSTEM_NAME from LMOD_SYSTEM_NAME or HOSTNAME"
  log error "Empty SYSTEM_NAME, needed to load environment and presets"
  exit 1
fi

# Check whether cmake/pre-commit change from environment
OLD_CMAKE="$(find_command cmake)"
OLD_PRE_COMMIT="$(find_command pre-commit)"
OLD_XDG_CACHE_HOME="${XDG_CACHE_HOME}"

# Load environment paths using the system name
ENV_SCRIPT="$(get_system_env "${SYSTEM_NAME}")"
if [ -n "${ENV_SCRIPT}" ]; then
  source_script "${ENV_SCRIPT}"
fi
CMAKE="$(find_command cmake)"

# Check whether the environment changed
needs_env=false
if [ "$(find_command cmake)" != "${OLD_PRE_COMMIT}" ]; then
  log warning "Local environment script uses a different pre-commit than your \$PATH"
  needs_env=true
fi
if [ "${CMAKE}" != "${OLD_CMAKE}" ]; then
  log warning "Local environment script uses a different CMake than your \$PATH"
  needs_env=true
fi
if [ "${OLD_XDG_CACHE_HOME}" != "${XDG_CACHE_HOME}" ]; then
  log warning "Cache directory changed from XDG_CACHE_HOME=${OLD_XDG_CACHE_HOME} to ${XDG_CACHE_HOME}"
  needs_env=true
fi
if ${needs_env}; then
  rc_file="$(get_shell_rc_file)"
  if install_shell_env "${rc_file}" "${ENV_SCRIPT}"; then
    needs_env=false
  else
    log warning "Please manually add the following to ${rc_file}:"
    printf ' . %s\n' "${ENV_SCRIPT}" >&2
  fi
fi

# Link preset file
ln_presets "${SYSTEM_NAME}"

# Search for and probe ccache
setup_ccache

# Check arguments and give presets if missing
if [ $# -eq 0 ] ; then
  printf '%s\n' "Usage: $0 PRESET [config_args...]" >&2
  if [ -n "${CMAKE}" ]; then
    log error "cmake unavailable: cannot call --list-presets"
    exit 1
  fi
  if "${CMAKE}" --list-presets >&2 ; then
    log info "Try using the '${SYSTEM_NAME}' or 'dev' presets ('base'), or 'default' if not"
  else
    log error "CMake may be too old or JSON file may be broken"
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
else
  log error "build failed: check configuration and build errors above"
fi

if ${needs_env}; then
  log warning "Environment changed: please manually add the following to ${rc_file}:"
  printf ' . %s\n' "${ENV_SCRIPT}" >&2
fi
