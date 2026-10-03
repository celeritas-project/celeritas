#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Build the Spack base image and the buildcache image with the Spack revisions
# pinned by .github/actions/setup-spack. Additional arguments are passed to
# each build command, e.g. --no-cache.
#
# Docker builds use BuildKit through the buildx plugin.
# Podman is used if Docker is unavailable or cannot
# reach its daemon (e.g. without root access). Set CONTAINER to override the
# container engine, e.g. CONTAINER=podman-hpc.
#-----------------------------------------------------------------------------#

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

# Check that docker can build with BuildKit and reach its daemon
check_docker() {
  if ! have docker; then
    return 1
  fi
  if ! docker buildx version >/dev/null 2>&1; then
    log warning "docker buildx is not installed: see https://docs.docker.com/go/buildx/"
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    log warning "cannot connect to the docker daemon"
    return 1
  fi
}

if [ -z "${CONTAINER}" ]; then
  if check_docker; then
    CONTAINER=docker
  elif have podman; then
    CONTAINER=podman
  else
    log error "neither docker (with buildx) nor podman is usable"
    exit 1
  fi
elif [ "${CONTAINER}" = docker ]; then
  if ! check_docker; then
    log error "docker is not usable"
    exit 1
  fi
elif ! have "${CONTAINER}"; then
  log error "${CONTAINER} is not available"
  exit 1
fi

if [ "${CONTAINER}" = docker ]; then
  # Load into the local image store, which the docker-container driver of
  # non-default buildx builders does not do implicitly
  BUILD="docker buildx build --load"
else
  # Podman defaults to the OCI format, which drops some Dockerfile metadata
  BUILD="${CONTAINER} build --format docker"
fi

pins=$(python3 "${SOURCE_DIR}/scripts/ci/parse-spack-versions.py" \
  "${SOURCE_DIR}/.github/actions/setup-spack/action.yml")
eval "${pins}"
TAG=$(printf "%.7s-%.7s" "${SPACK_REF}" "${SPACK_PACKAGES_REF}")

build() {
  target=$1
  shift
  ${BUILD} \
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
