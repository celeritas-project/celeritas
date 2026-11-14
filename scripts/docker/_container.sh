#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Shared setup for the docker image build scripts: source (do not execute)
# this file after setting CELER_SOURCE_DIR. It defines:
#   log LEVEL MESSAGE   print a message to stderr
#   CONTAINER_EXE       container engine (docker or podman-like)
#   BUILD               image build command
#   SPACK_REPO, SPACK_REF, SPACK_PACKAGES_REPO, SPACK_PACKAGES_REF
#                       Spack revisions pinned by the setup-spack CI action
#
# Docker builds use BuildKit through the buildx plugin.
# Podman is used if Docker is unavailable or cannot
# reach its daemon (e.g. without root access).
# Set CONTAINER_EXE to override the container engine, e.g. CONTAINER_EXE=podman-hpc.
#-----------------------------------------------------------------------------#

if command -v celerlog >/dev/null 2>&1; then
  log() {
    celerlog "$@"
  }
else
  log() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi

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

if [ -z "${CONTAINER_EXE}" ]; then
  if check_docker; then
    CONTAINER_EXE=docker
  elif have podman; then
    CONTAINER_EXE=podman
  else
    log error "neither docker (with buildx) nor podman is usable"
    exit 1
  fi
elif [ "${CONTAINER_EXE}" = docker ]; then
  if ! check_docker; then
    log error "docker is not usable"
    exit 1
  fi
elif ! have "${CONTAINER_EXE}"; then
  log error "${CONTAINER_EXE} is not available"
  exit 1
fi

if [ "${CONTAINER_EXE}" = docker ]; then
  # Load into the local image store, which the docker-container driver of
  # non-default buildx builders does not do implicitly
  BUILD="docker buildx build --load"
else
  # Podman defaults to the OCI format, which drops some Dockerfile metadata
  BUILD="${CONTAINER_EXE} build --format docker"
fi

GHA_FILE="${CELER_SOURCE_DIR}/.github/actions/setup-spack/action.yml"
pins=$(python3 "${CELER_SOURCE_DIR}/scripts/ci/parse-spack-versions.py" "${GHA_FILE}")
eval "${pins}"
log info "Using Spack ${SPACK_REF} and packages ${SPACK_PACKAGES_REF} from ${GHA_FILE}"
