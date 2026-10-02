#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Check that the Spack revisions built into the image match the ones pinned by
# the setup-spack CI action of the Celeritas source at CELER_SOURCE_DIR, then
# run a command.
#-----------------------------------------------------------------------------#

set -e
log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

ACTION_FILE="${CELER_SOURCE_DIR}/.github/actions/setup-spack/action.yml"
if [ ! -f "${ACTION_FILE}" ]; then
  log error "${ACTION_FILE} not found: mount a Celeritas checkout at ${CELER_SOURCE_DIR}"
  exit 1
fi

pins=$(python3 "$(dirname "$0")/spack-pins.py" "${ACTION_FILE}")
eval "${pins}"
if [ "${SPACK_REF}" != "${CELER_SPACK_REF}" ] \
    || [ "${SPACK_PACKAGES_REF}" != "${CELER_SPACK_PACKAGES_REF}" ]; then
  log error "Spack revisions in the image do not match ${ACTION_FILE}:"
  log error "  spack:          image ${CELER_SPACK_REF}, source ${SPACK_REF}"
  log error "  spack-packages: image ${CELER_SPACK_PACKAGES_REF}, source ${SPACK_PACKAGES_REF}"
  log error "rebuild the image with scripts/docker/buildcache/build.sh"
  exit 1
fi

exec "$@"
