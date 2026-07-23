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

export SPACK_ROOT="/cvmfs/dune.opensciencegrid.org/spack/v1.1.1"
SPACK_ENV_NAME="dunesw-10_21_01d00-justin-01_06_01-prototype"

# Remove cuda home and compiler from parent (milan2) environment
unset CUDA_HOME
unset CUDACXX

celerlog info "Setting up spack environment from ${SPACK_ROOT}"
. "${SPACK_ROOT}/share/spack/setup-env.sh"
celerlog info "Loading from spack environment '${SPACK_ENV_NAME}'"
_spack_src_file=$(mktemp -p ${SCRATCHDIR}/build spack-XXXXXX.sh)
command spack -e ${SPACK_ENV_NAME} load --sh \
  gcc cmake root art larsim googletest cuda \
  > ${_spack_src_file}
celerlog debug "Temporary spack environment setup script: ${_spack_src_file}"
. ${_spack_src_file}
if [ ! command -v lar >/dev/null 2>&1 ]; then
  celerlog error "failed to load spack environment: see ${_spack_src_file}"
  return 1
fi

#-----------------------------------------------------------------------------#

if [ -n "$CELER_SOURCE_DIR" ]; then
  _clangd="$CELER_SOURCE_DIR/.clangd"
  if [ ! -e "${_clangd}" ]; then
    # Create clangd compatible with the system and build config
    if [ ! -x "${CXX}" ]; then
      celerlog info "GCC isn't loaded as expected at \$CXX = ${CXX}"
    else
      celerlog info "Creating clangd config using ${CXX}: ${_clangd}"

      # Extract include paths from GCC
      _gcc_includes=$("${CXX}" -E -x c++ - -v < /dev/null 2>&1 | \
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
