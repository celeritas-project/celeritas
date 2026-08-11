#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# Allow running from user rc setup outside of build.sh environment
if ! command -v celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi

#-----------------------------------------------------------------------------#
# Call this helper function on the login node bare metal
_apptainer_fnal() {
  if ! [ -d "${SCRATCHDIR}" ]; then
    celerlog error "Scratch directory does not exist: run
  . \${CELER_SOURCE}/scripts/env/${SYSTEM_NAME:-excl}.sh
"
    return 1
  fi

  if ! [ -d "/cvmfs" ]; then
    celerlog error "cannot run apptainer image: CVMFS is not available on this host"
    return 1
  fi

  # BEGIN_DOC_APPTAINER
  APPTAINER_DIR=/usr
  IMAGE_DIR=/cvmfs/singularity.opensciencegrid.org/fermilab
  IMAGE=${1:-fnal-dev-el9:devel}
  exec $APPTAINER_DIR/bin/apptainer \
    shell --shell=/bin/bash \
    -B /cvmfs,$SCRATCHDIR,${HOME},${CELER_APPTAINER_FWD}, \
    --nv --ipc --pid  \
    ${IMAGE_DIR}/${IMAGE}
  # END_DOC_APPTAINER
}
alias apptainer-fnal=_apptainer_fnal
# END APPTAINER SCRIPT

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

# Set scratchdir: /scratch should exist according to excl docs
if ! [ -d "/scratch" ]; then
  celerlog error "Scratch directory does not exist at '/'"
  return 1
fi
export SCRATCHDIR="/scratch/$USER"
if [ -n "${APPTAINER_NAME}" ]; then
  export SCRATCHDIR="${SCRATCHDIR}/${APPTAINER_NAME%%:*}"
fi

for _d in cache build install ; do
  # Create build/install in higher-performance local-but-persistent dir
  _scratch="$SCRATCHDIR/$_d"
  if ! test -d "${_scratch}"; then
    celerlog info "Creating scratch directory at ${_scratch}"
    mkdir -p "${_scratch}" || return $?
    chmod 700 "${_scratch}" || return $?
  fi
  unset _scratch
done
export XDG_CACHE_HOME="${SCRATCHDIR}/cache"

if [ -n "${APPTAINER_NAME}" ]; then
  # Only set up spack and environment on bare metal
  celerlog debug "Skipping excl spack environment setup and cleaning vars: running inside apptainer"
  unset CC CXX CMAKE_PREFIX_PATH

  # Override apptainer command
  _apptainer_fnal() {
    printf "error: %s\n" "cannot run apptainer inside ${APPTAINER_NAME}"
    return 1
  }
  return 0
fi

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

CELER_SCRATCHDIR=/scratch/celeritas
export CELER_APPTAINER_FWD=${CELER_SCRATCHDIR},/auto/projects/celeritas/spack-cache


CELER_SPACK_ENV="celeritas-${SYSTEM_NAME:-excl}-scratch"
CELER_SPACK_VIEW=${CELER_SCRATCHDIR}/view
if ! [ -d "${CELER_SPACK_VIEW}" ]; then
  celerlog error "Celeritas spack environment does not exist (or is unreadable) at CELER_SPACK_VIEW=${CELER_SPACK_VIEW}"
  return 1
fi

# CELER_SPACK_OPT can be used by downstream env for exact paths, e.g. CUDA
# it is exported to make it available to subshells
export CELER_SPACK_OPT=${CELER_SCRATCHDIR}/opt/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder
if ! [ -d "${CELER_SPACK_OPT}" ]; then
  celerlog warning "Celeritas toolchain does not exist (or is unreadable) at CELER_SPACK_OPT=${CELER_SPACK_OPT}"
fi

export PATH=${CELER_SPACK_VIEW}/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELER_SPACK_VIEW}:${CMAKE_PREFIX_PATH}
