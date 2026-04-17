#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if [ -z "${APPTAINER_CONTAINER}" ]; then
  celerlog error "Not running in an apptainer"
  return 1
fi

if ! [ -d "/cvmfs" ]; then
  celerlog error "CVMFS is not mounted"
  return 1
fi

if [ -z "${SCRATCHDIR}" ]; then
  celerlog error "\$SCRATCHDIR is not defined"
  return 1
fi

if [ ! -z "${CMAKE_PREFIX_PATH}" ]; then
  celerlog warning "Existing CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} may interfere with build"
fi

#-----------------------------------------------------------------------------#

celerlog info "Running in apptainer ${APPTAINER_CONTAINER}"
if [ -z "${MRB_PROJECT}" ]; then
  export MRB_PROJECT=larsoft
  # NOTE: 10.14 uses Geant4 10.6.1, and 10.20 uses 11.2
  # export MRB_PROJECT_VERSION=v10_14_01
  export MRB_PROJECT_VERSION=v10_20_01
fi
if [ -z "${MRB_QUALS}" ]; then
  export MRB_QUALS=e26:prof
fi

if [ -n "${UPS_DIR}" ]; then
  celerlog debug "UPS already set up: ${UPS_DIR}"
else
  celerlog info "Setting up DUNE UPS"
  . /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
  celerlog debug "Using UPS_OVERRIDE=${UPS_OVERRIDE}, MRB_PROJECT=${MRB_PROJECT}"
fi
if [ -n "${SETUP_LARSOFT}" ]; then
  celerlog debug "LARSOFT is already set up"
else
  # Set up larsoft build defaults with UPS
  celerlog info "Setting up ${MRB_PROJECT} ${MRB_PROJECT_VERSION} with qualifiers '${MRB_QUALS}'"
  setup ${MRB_PROJECT} ${MRB_PROJECT_VERSION} -q ${MRB_QUALS} || return $?
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
