#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# BEGIN APPTAINER SCRIPT
# Call this helper function on the login node bare metal
apptainer_fermilab() {
  if ! [ -d "${SCRATCHDIR}" ]; then
    echo "Scratch directory does not exist: run
  . \${CELERITAS_SOURCE}/scripts/env/scisoftbuild01.sh
"
    return 1
  fi

  # Use the Scientific Linux distribution required for the dependencies
  image=${1:-fnal-dev-sl7:latest}

  # Start apptainer, forwarding necessary directories
  MIS_DIR=/cvmfs/oasis.opensciencegrid.org/mis # codespell:ignore
  exec $MIS_DIR/apptainer/current/bin/apptainer \
    shell --shell=/bin/bash \
    -B /cvmfs,$SCRATCHDIR,$HOME,$XDG_RUNTIME_DIR,/opt,/etc/hostname,/etc/hosts,/etc/krb5.conf  \
    --ipc --pid  \
    /cvmfs/singularity.opensciencegrid.org/fermilab/${image}
}
# END APPTAINER SCRIPT

#-----------------------------------------------------------------------------#

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

# Allow running from user rc setup outside of build.sh environment
if ! command -v celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi
if ! command -v setup  >/dev/null 2>&1; then
  celerlog debug "Missing 'setup' command: using local UPS"
  setup ()
  {
      . `ups setup "$@"`
  }
fi
if [ -z "${SYSTEM_NAME}" ]; then
  SYSTEM_NAME=$(uname -s)
  celerlog debug "Set SYSTEM_NAME=${SYSTEM_NAME}"
fi

# Set default scratch directory
export SCRATCHDIR="${SCRATCHDIR:-/scratch/$USER}"
for _d in build install cache; do
  # Create build/install in higher-performance local-but-persistent dir
  _scratch="$SCRATCHDIR/$_d"
  if ! [ -d "${_scratch}" ]; then
    celerlog info "Creating scratch directory '${_scratch}'"
    mkdir -p "${_scratch}" || return $?
    chmod 700 "${_scratch}" || return $?
  fi
  unset _scratch
done
export XDG_CACHE_HOME="${SCRATCHDIR}/cache"

if [ -n "${APPTAINER_CONTAINER}" ]; then
  export MRB_PROJECT=larsoft
  export MRB_PROJECT_VERSION=v10_14_01
  export MRB_QUALS=e26:prof
  celerlog info "Running in apptainer ${APPTAINER_CONTAINER}"
  if [ -n "${UPS_DIR}" ]; then
    celerlog debug "Dune UPS already set up: ${UPS_DIR}"
  else
    celerlog info "Setting up DUNE UPS"
    . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
    celerlog debug "Using UPS_OVERRIDE=${UPS_OVERRIDE}, MRB_PROJECT=${MRB_PROJECT}"
  fi
  if [ -n "${SETUP_LARCORE}" ]; then
    celerlog debug "LARCORE is already set up"
  else
    # Set up larsoft build defaults with UPS
    celerlog info "Setting up ${MRB_PROJECT} ${MRB_PROJECT_VERSION} with qualifiers '${MRB_QUALS}'"
    setup ${MRB_PROJECT} ${MRB_PROJECT_VERSION} -q ${MRB_QUALS} || return $?
  fi
fi

# Set up additional tools if running inside an apptainer
if [ -n "${MRB_PROJECT}" ]; then
  # Do not set up MRB: instead, just load cmake and cetmodules
  # (larsoft runtime dependencies have already been loaded)
  # Note that these do not need MRB_QUALS since they're not binary products
  setup cmake v3_27_4  || return $?
  setup cetmodules v3_24_01 || return $?
fi

if [ -n "$CELER_SOURCE_DIR" ]; then
  _clangd="$CELER_SOURCE_DIR/.clangd"
  if [ ! -e "${_clangd}" ]; then
    # Create clangd compatible with the system and build config
    _cxx=$GCC_FQ_DIR/bin/g++
    if [ ! -x "${_cxx}" ]; then
      celerlog info "GCC isn't loaded as expected at \$GCC_FQ_DIR/bin/g++ = ${_cxx}"
    else
      celerlog info "Creating clangd config using ${_cxx}: ${_clangd}"

      # Extract include paths from GCC
      _gcc_includes=$("${_cxx}" -E -x c++ - -v < /dev/null 2>&1 | \
        sed -n '/^#include <...> search starts here:/,/^End of search list\./p' | \
        grep '^ ' | sed 's/^ *//' | \
        awk '{printf "      -isystem,\n      %s,\n", $0}' | \
        sed '$s/,$//')

      cat > "${_clangd}" << EOF
CompileFlags:
  CompilationDatabase: ${SCRATCHDIR}/build/celeritas-reldeb-orange
  Add:
    [
${_gcc_includes}
    ]
EOF
      unset _gcc_includes
    fi
    unset _clangd
  fi
fi
