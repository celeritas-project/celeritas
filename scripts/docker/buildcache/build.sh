#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Build the Spack base image and the buildcache image with the Spack revisions
# pinned by .github/actions/setup-spack. Additional arguments are passed to
# each build command, e.g. --no-cache. Set DOCKER to override the container
# engine, e.g. DOCKER=podman-hpc.
#-----------------------------------------------------------------------------#

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

BUILDARGS=
if [ -z "${DOCKER}" ]; then
  DOCKER=docker
  if ! command -v ${DOCKER} >/dev/null 2>&1; then
    DOCKER=podman
    BUILDARGS="--format docker"
  fi
fi
if ! command -v ${DOCKER} >/dev/null 2>&1; then
  echo "error: ${DOCKER} is not available" >&2
  exit 1
fi

pins=$(python3 "${SOURCE_DIR}/scripts/ci/parse-spack-versions.py" \
  "${SOURCE_DIR}/.github/actions/setup-spack/action.yml")
eval "${pins}"
TAG=$(printf "%.7s-%.7s" "${SPACK_REF}" "${SPACK_PACKAGES_REF}")

build() {
  target=$1
  shift
  ${DOCKER} build ${BUILDARGS} \
    --build-arg SPACK_REF="${SPACK_REF}" \
    --build-arg SPACK_PACKAGES_REF="${SPACK_PACKAGES_REF}" \
    --target "${target}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "$@" \
    "${SCRIPT_DIR}"
}

build spack-base -t "celeritas/spack-ubuntu24:${TAG}" "$@"
build buildcache -t "celeritas-buildcache:${TAG}" \
  -t celeritas-buildcache:latest "$@"
