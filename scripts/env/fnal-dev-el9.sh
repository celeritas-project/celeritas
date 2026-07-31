#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

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

if [ -n "${APPTAINER_CONTAINER}" ]; then
  celerlog info "Running in apptainer ${APPTAINER_CONTAINER}"
fi

# BEGIN_DOC_FNALSPACK
# Latest release of FNAL-Spack and DUNESW spack environment
export SPACK_ROOT="/cvmfs/dune.opensciencegrid.org/spack/v1.1.1"
CELER_SPACK_ENV="dunesw-10_21_01d00-justin-01_06_01-prototype"
CELER_SPACK_PACKAGES="gcc cmake larsim googletest cuda"
# END_DOC_FNALSPACK

# Remove cuda home and compiler from parent (milan2) environment
# since these interfere with the build
unset CUDA_HOME
unset CUDACXX

# Try loading spack commands
celerlog info "Setting up spack environment from ${SPACK_ROOT}"
. "${SPACK_ROOT}/share/spack/setup-env.sh"
_errcode=$?
if [ "${_errcode}" -ne 0 ]; then
  celerlog error "Failed to set up spack"
  return ${_errcode}
fi

# Set up the environment variables necessary to load Celeritas build requirements.
# We do this rather than loading the entire environment because:
# 1. The environment contains a broken build of googletest (preventing testing)
# 2. Builds should be faster and safer due to fewer packages in the environment directories
# 3. Setup is quicker, and saving to a `.sh` file dramatically reduces time for subsequent rebuilds.
#
# NOTE that this environment is *not sufficient* to run `lar`; it is only used to build and test Celeritas.
celerlog info "Loading from spack environment '${CELER_SPACK_ENV}'"
_spack_src_file="${SCRATCHDIR}/build/spack-env.sh"


if ! [ -f "${_spack_src_file}" ]; then
  # Create a cached environment setup script
  _tmp_src_file=$(mktemp ${_spack_src_file}.XXXXXX)
  command spack -e "${CELER_SPACK_ENV}" load --sh \
    ${CELER_SPACK_PACKAGES} \
    > ${_tmp_src_file}
  _errcode=$?
  if [ ${_errcode} -ne 0 ]; then
    celerlog error "Failed to create spack environment at ${_tmp_src_file}"
    return ${_errcode}
  fi
  # Prevent CMake from removing `-I` from build lines due to the C include path being set at configure time
  # (this results in missing TBB includes when rebuilding if the `.sh` file isn't sourced)
  printf "\n%s\n" \
    "unset C_INCLUDE_PATH" \
    >> ${_tmp_src_file}
  mv "${_tmp_src_file}" "${_spack_src_file}"
  celerlog info "Created spack environment setup script: ${_spack_src_file}"
else
  celerlog info "Reusing spack environment setup script at ${_spack_src_file}"
fi

# Load the build environment
. "${_spack_src_file}"
_errcode=$?
if [ "${_errcode}" -ne 0 ]; then
  celerlog error "Failed to source spack environment"
  mv "${_spack_src_file}" "${_spack_src_file}.old"
  return ${_errcode}
fi

if [ ! command -v lar >/dev/null 2>&1 ]; then
  celerlog error "Incorrect spack environment: see ${_spack_src_file}.old"
  mv "${_spack_src_file}" "${_spack_src_file}.old"
  return 1
fi
unset _errcode
