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

if ! [ -d "/cvmfs" ]; then
  celerlog error "CVMFS is not mounted"
  return 1
fi

if [ -z "${SCRATCHDIR}" ]; then
  celerlog error "\$SCRATCHDIR is not defined"
  return 1
fi

#-----------------------------------------------------------------------------#
# Helper function to set up LArSoft with Celeritas
_setup_larsoft_celer() {
  # BEGIN_DOC_LARENV
  # Load spack
  if command -v spack >/dev/null 2>&1; then
    celerlog debug "Loading spack commands"
    source "${SPACK_ROOT}/share/spack/setup-env.sh"
  fi
  # Source DUNESW environment
  if [ -z "${SPACK_ENV}" ]; then
    celerlog debug "Activating spack environment"
    spack env activate "${CELER_SPACK_ENV}"
  fi
  # Check for celeritas plugin (user must load themselves)
  if [ -z "${CELERITAS_DIR}" ]; then
    celerlog warning "Celeritas not loaded"
    celerlog info "Run: eval \$(\${CELERITAS_DIR}/bin/larceler-env)"
  fi
  # Set up local FHICL/GDML paths
  if ! printf '%s\n' "${FW_SEARCH_PATH}" | grep -Eq '(^|:)[.](:|$)'; then
    export FW_SEARCH_PATH=".:${FW_SEARCH_PATH}"
    export FHICL_FILE_PATH=".:./job:${FHICL_FILE_PATH}"
  fi
  # END_DOC_LARENV

  if ! command -v lar >/dev/null 2>&1; then
    celerlog error "lar not found: environment failed to set up"
    return 1
  fi
}
alias setup-larsoft-celer=_setup_larsoft_celer

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

# Remove cuda home and compiler in case they exist in parent environment
# (incompatible versions or inaccessible locations may interfere with build)
unset CUDA_HOME
unset CUDACXX

if ! command -v spack >/dev/null 2>&1 ; then
  # Load spack shell commands and path
  celerlog info "Setting up spack environment from ${SPACK_ROOT}"
  . "${SPACK_ROOT}/share/spack/setup-env.sh"
  _errcode=$?
  if [ "${_errcode}" -ne 0 ]; then
    celerlog error "Failed to set up spack"
    return ${_errcode}
  fi
fi

# Set up the environment variables necessary to load Celeritas build requirements.
# We do this rather than loading the entire environment because:
# 1. The environment contains a broken build of googletest (preventing testing)
# 2. Builds should be faster and safer due to fewer packages in the environment directories
# 3. Setup is quicker, and saving to a `.sh` file dramatically reduces time for subsequent rebuilds.
#
# NOTE that this environment is *not sufficient* to run `lar`; it is only used to build and test Celeritas.
_spack_src_file="${SCRATCHDIR}/build/spack-env.sh"

if ! [ -f "${_spack_src_file}" ]; then
  # Create a cached environment setup script
  celerlog info "Loading spack environment '${CELER_SPACK_ENV}' packages: ${CELER_SPACK_PACKAGES}"
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
  # Also note the environment that we loaded
  printf "\n%s\n%s\n" \
    "unset C_INCLUDE_PATH" \
    "export CELER_SPACK_ENV_LOADED=${CELER_SPACK_ENV}" \
    >> ${_tmp_src_file}
  mv "${_tmp_src_file}" "${_spack_src_file}"
else
  celerlog debug "Spack environment setup script already exists at '${_spack_src_file}'"
fi

if [ -z "${CELER_SPACK_ENV_LOADED}" ]; then
  celerlog info "Loading spack build environment from '${_spack_src_file}'"
  # Load the build environment
  . "${_spack_src_file}"
  _errcode=$?
  if [ "${_errcode}" -ne 0 ]; then
    celerlog error "Failed to source spack environment"
    mv "${_spack_src_file}" "${_spack_src_file}.old"
    return ${_errcode}
  fi
elif [ "${CELER_SPACK_ENV}" != "${CELER_SPACK_ENV_LOADED}" ]; then
  celerlog warning "Loaded environment '${CELER_SPACK_ENV_LOADED}' does not match expected '${CELER_SPACK_ENV}'"
  celerlog warning "Skipping spack build environment: '${_spack_src_file}'"
else
  celerlog info "Spack build environment ${CELER_SPACK_ENV_LOADED} already loaded"
fi

if [ ! command -v lar >/dev/null 2>&1 ]; then
  celerlog error "Incorrect spack environment: see ${_spack_src_file}.old"
  mv "${_spack_src_file}" "${_spack_src_file}.old"
  return 1
fi
unset _errcode
