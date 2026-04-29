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

#-----------------------------------------------------------------------------#
# Set up environment

celerlog info "Running in apptainer ${APPTAINER_CONTAINER}"

if [ -n "${UPS_DIR}" ]; then
  celerlog debug "UPS already set up: ${UPS_DIR}"
else
  if [ ! -z "${CMAKE_PREFIX_PATH}" ]; then
    celerlog warning "Existing CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} may interfere with build"
  fi

  celerlog info "Setting up LArSoft"
  # BEGIN_DOC_UPS
  . /cvmfs/larsoft.opensciencegrid.org/setup_larsoft.sh
  setup larsoft v10_20_01 -q e26:prof
  setup cmake v3_27_4
  setup cetmodules v3_24_01
  # END_DOC_UPS
fi

#-----------------------------------------------------------------------------#

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
