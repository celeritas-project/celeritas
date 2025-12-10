#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
#-----------------------------------------------------------------------------#
# Note: to launch the apptainer run this
apptainer_fermilab() {
  if ! [ -d "${SCRATCHDIR}" ]; then
    echo "Scratch directory does not exist: ru
  . scripts/env/scisoftbuild01.sh
"
    return 1
  fi

  image=${1:-fnal-dev-sl7:latest}

  # Start apptainer, forwarding necessary directories
  SUBDOMAIN=mis # codespell:ignore
  exec /cvmfs/oasis.opensciencegrid.org/$SUBDOMAIN/apptainer/current/bin/apptainer \
    shell --shell=/bin/bash \
    -B /cvmfs,$SCRATCHDIR,$HOME,$XDG_RUNTIME_DIR,/opt,/etc/hostname,/etc/hosts,/etc/krb5.conf  \
    --ipc --pid  \
    /cvmfs/singularity.opensciencegrid.org/fermilab/${image}
}

#-----------------------------------------------------------------------------#

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

# Allow running from user rc setup outside of build.sh environment
if ! command -v celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi
if [ -z "${SYSTEM_NAME}" ]; then
  SYSTEM_NAME=$(uname -s)
  celerlog debug "Set SYSTEM_NAME=${SYSTEM_NAME}"
fi

if [ -n "${APPTAINER_CONTAINER}" ]; then
  export MRB_PROJECT=larsoft
  export MRB_PROJECT_VERSION=v10_14_01
  export MRB_QUALS=e26:prof
  celerlog info "Running in apptainer ${APPTAINER_CONTAINER}"
  if [ -z "${UPS_DIR}" ]; then
    celerlog info "Setting up DUNE UPS"
    . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
    celerlog debug "Using UPS_OVERRIDE=${UPS_OVERRIDE}, MRB_PROJECT=${MRB_PROJECT}"
  fi
  if [ -z "${SETUP_LARCORE}" ]; then
    # Set up larsoft build defaults with UPS
    celerlog info "Setting up ${MRB_PROJECT} ${MRB_PROJECT_VERSION} with qualifiers '${MRB_QUALS}'"
    setup ${MRB_PROJECT} ${MRB_PROJECT_VERSION} -q ${MRB_QUALS} || return $?
  fi
fi

# Set default scratchdir; /scratch should exist according to excl docs
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

# Set up larsoft if running inside an apptainer
if [ -n "${MRB_PROJECT}" ]; then
  LARSCRATCHDIR="${SCRATCHDIR}/${MRB_PROJECT}"
  if ! [ -d "${LARSCRATCHDIR}" ]; then
    celerlog info "Creating MRB dev area in ${LARSCRATCHDIR}..."
    mkdir -p "${LARSCRATCHDIR}" || return $?
    (
      cd "${LARSCRATCHDIR}"
      mrb newDev
    )
    celerlog debug "MRB environment created"
  fi
  _setup_filename="${LARSCRATCHDIR}/localProducts_${MRB_PROJECT}_${MRB_PROJECT_VERSION}_${MRB_QUALS//:/_}/setup"
  if ! [ -f "${_setup_filename}" ]; then
    celerlog warn "Expected setup file at ${_setup_filename}: MRB may not have been set up correctly"
    _setup_filename=$(print %s"${LARSCRATCHDIR}/localProducts_${MRB_PROJECT}*/setup")
    if [ -f "${_setup_filename}" ]; then
      celerlog info "Found setup file ${_setup_filename}"
    fi
  fi
  . "${_setup_filename}"
fi

# Check out a package so that mrb will load cmake (may be arbitrary?)
if [ -n "${MRB_SOURCE}" ]; then
  _pkg=larsim
  if ! [ -d "${MRB_SOURCE}/${_pkg}" ]; then
    celerlog info "Installing ${_pkg}"
    mrb g ${_pkg}
  fi

  # Now that a package exists in MRB source, cmake and dependencies can load
  celerlog info "Activating MRB environment"
  mrbsetenv
  celerlog debug "MRB setup complete"
fi

if [ -n "$CELER_SOURCE_DIR" ]; then
  _clangd="$CELER_SOURCE_DIR/.clangd"
  if [ ! -e "${_clangd}" ]; then
    # Create clangd compatible with the system and build config
    _gcc_version=$(gcc -dumpversion | cut -d. -f1)
    celerlog info "Creating clangd config using GCC ${_gcc_version}: ${_clangd}"
    cat > "${_clangd}" << EOF
CompileFlags:
  CompilationDatabase: ${SCRATCHDIR}/build/celeritas-reldeb-orange
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

export XDG_CACHE_HOME="${SCRATCHDIR}/cache"
