#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

# Allow running from user rc setup outside of build.sh environment
if ! command -v celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi
if test -z "${SYSTEM_NAME}"; then
  SYSTEM_NAME=$(hostname -s)
  celerlog debug "Set SYSTEM_NAME=${SYSTEM_NAME}"
fi

export SPACK_ROOT=/auto/projects/celeritas/spack
if ! test -r $SPACK_ROOT; then
  celerlog error "spack directory SPACK_ROOT=${SPACK_ROOT} is not readable:
       contact excl-help@ornl.gov for access"
  exit 1
fi

if ! command -v spack >/dev/null 2>&1; then
  _spack_setup="$SPACK_ROOT/share/spack/setup-env.sh"
  case "$0" in
    /bin/*)
      # Presumably sourcing from user shell to set up environment: load spack
      celerlog debug "Loading spack setup from ${_spack_setup}"
      . "${_spack_setup}"
      ;;
    *)
      # Spack isn't available but we're loading from the build script so we don't really need it
      celerlog warning "spack is not available; source ${_spack_setup} if desired"
      ;;
  esac
  unset _spack_setup
fi

# Set default scratchdir; /scratch should exist according to excl docs
export SCRATCHDIR="${SCRATCHDIR:-/scratch/$USER}"
for _d in build install cache; do
  # Create build/install in higher-performance local-but-persistent dir
  _scratch="$SCRATCHDIR/$_d"
  if ! test -d "${_scratch}"; then
    celerlog info "Creating scratch directory at ${_scratch}"
    mkdir -p "${_scratch}" || return $?
    chmod 700 "${_scratch}" || return $?
  fi
  unset _scratch
done

if [ -n "$CELER_SOURCE_DIR" ]; then
  _clangd="$CELER_SOURCE_DIR/.clangd"
  if [ ! -e "${_clangd}" ]; then
    # Create clangd compatible with the system and build config
    _gcc_version=$(gcc -dumpversion | cut -d. -f1)
    celerlog info "Creating clangd config using GCC ${_gcc_version}: ${_clangd}"
    cat > "${_clangd}" << EOF
CompileFlags:
  CompilationDatabase: ${SCRATCHDIR}/build/celeritas-reldeb
  Add:
    [
      -isystem,
      /usr/include/c++/${_gcc_version},
      -isystem,
      /usr/local/include,
      -isystem,
      /usr/include,
      -isystem,
      /usr/include/x86_64-linux-gnu/c++/${_gcc_version},
    ]
EOF
  fi
  unset _clangd
fi

CELERITAS_ENV=${SPACK_ROOT}/var/spack/environments/celeritas-${SYSTEM_NAME}/.spack-env/view

if ! [ -e "${CELERITAS_ENV}" ]; then
  celerlog error "celeritas env does not exist (or is unreadable) at ${CELERITAS_ENV}"
  return 1
fi

export PATH=${CELERITAS_ENV}/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELERITAS_ENV}:${CMAKE_PREFIX_PATH}

export XDG_CACHE_HOME="${SCRATCHDIR}/cache"
