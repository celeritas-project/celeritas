#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Build the Spack base image and the buildcache image with the Spack revisions
# pinned by .github/actions/setup-spack. Additional arguments are passed to
# each build command, e.g. --no-cache.
# See scripts/docker/_container.sh for the container engine selection.
#-----------------------------------------------------------------------------#

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd) # NOT the scripts/ci dir
CELER_SOURCE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

. "${CELER_SOURCE_DIR}/scripts/docker/_container.sh"
TAG=$(printf "%.7s-%.7s" "${SPACK_REF}" "${SPACK_PACKAGES_REF}")
log info "Set tag ${TAG}"

PROJECT=celeritas

build() {
  target=$1
  shift
  ${BUILD} \
    --build-arg SPACK_REPO="${SPACK_REPO}" \
    --build-arg SPACK_REF="${SPACK_REF}" \
    --build-arg SPACK_PACKAGES_REPO="${SPACK_PACKAGES_REPO}" \
    --build-arg SPACK_PACKAGES_REF="${SPACK_PACKAGES_REF}" \
    --target "${target}" \
    -t "${PROJECT}/${target}:${TAG}" \
    -t "${PROJECT}/${target}:latest" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "$@" \
    "${SCRIPT_DIR}"
}

# Build target images inside Dockerfile
build spack-base-ubuntu24 "$@"
build buildcache "$@"
