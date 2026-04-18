#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! command -v load_system_env >/dev/null 2>&1; then
  printf "error: expected load_system_env helper function via build.sh or shell\n" >&2
  return 1
fi

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

# Call this helper function on the login node bare metal
_apptainer_fnal() {
  if ! [ -d "${SCRATCHDIR}" ]; then
    echo "Scratch directory does not exist: run
  . \${CELERITAS_SOURCE}/scripts/env/scisoftbuild01.sh
"
    return 1
  fi

  # BEGIN_DOC_APPTAINER
  APPTAINER_DIR=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current # codespell:ignore
  IMAGE_DIR=/cvmfs/singularity.opensciencegrid.org/fermilab
  IMAGE=fnal-dev-sl7:latest
  exec $APPTAINER_DIR/bin/apptainer \
    shell --shell=/bin/bash \
    -B /cvmfs,$SCRATCHDIR,$HOME,$XDG_RUNTIME_DIR,/opt,/etc/hostname,/etc/hosts,/etc/krb5.conf  \
    --ipc --pid  \
    ${IMAGE_DIR}/${IMAGE}
  # END_DOC_APPTAINER
}
alias apptainer-fnal=_apptainer_fnal

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

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
