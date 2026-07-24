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

# Call this helper function on the login node bare metal
_apptainer_fnal() {
  if ! [ -d "${SCRATCHDIR}" ]; then
    echo "Scratch directory does not exist: run
  . \${CELER_SOURCE}/scripts/env/${SYSTEM_NAME:-scisoftbuild01}.sh
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

# Set scratchdir: /scratch should exist on scisoftbuild
if ! [ -d "/scratch" ]; then
  celerlog error "Scratch directory does not exist at '/'"
  return 1
fi
export SCRATCHDIR="/scratch/$USER"
if [ -n "${APPTAINER_NAME}" ]; then
  export SCRATCHDIR="${SCRATCHDIR}/${APPTAINER_NAME%%:*}"
fi

for _d in cache build install; do
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

# Prevent Celeritas tests from trying to use nonexistent CUDA device, even though we build with it
export CELER_DISABLE_DEVICE=1

if [ -z "${APPTAINER_NAME}" ]; then
  # Check that we're using AlmaLinux 9
  if grep -q "platform:el9" /etc/os-release ; then
    celerlog debug "Loading fnal-dev-el9 environment"
    # NOTE: setting SYSTEM_NAME changes the linked cmake presets inside build.sh
    SYSTEM_NAME=fnal-dev-el9
    load_system_env ${SYSTEM_NAME}
  fi
fi
