#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Build the GPU development and CI images with the Spack revisions pinned by
# .github/actions/setup-spack and tag them with the date in REGISTRY
# (default: ghcr.io/celeritas-project). The CI image is pushed only if PUSH=1
# is set. SPACK_INSTALL_ARGS (e.g. "-p 8 -j 32") is
# passed to 'spack install'.
# The images are labeled with the Celeritas git revision: set REVISION when
# building from a source directory that is not a git checkout.
# Additional arguments are passed to each build command, e.g. --no-cache.
# See scripts/docker/_container.sh for the container engine selection.
#-----------------------------------------------------------------------------#

if [ -z "$1" ]; then
  echo "usage: $0 {cuda|rocm} [build args...]" >&2
  exit 2
fi
CONFIG=$1
shift

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CELER_SOURCE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

. "${SCRIPT_DIR}/_container.sh"

# The base image sets the CUDA or ROCm version. When updating it, also update
# the versions listed in README.md and the image tags in
# .github/workflows/build-docker.yml
case ${CONFIG} in
  cuda)
    BASE_IMAGE=docker.io/nvidia/cuda:13.5.0-devel-ubuntu24.04
    ;;
  rocm)
    BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4
    # Only the libraries Celeritas uses: the "complete" image is 6 GB larger
    GPU_APT_PACKAGES="hipcub-dev hiprand-dev rocprim-dev rocrand-dev rocthrust-dev roctracer-dev"
    ;;
  *)
    log error "invalid configuration: ${CONFIG}"
    exit 1
    ;;
esac

# Use UTC so the tag does not depend on the build host's time zone
CREATED=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
DATE=${CREATED%%T*}

# Full commit hash of the source, with a suffix for uncommitted changes
if [ -z "${REVISION}" ]; then
  REVISION=$(git -C "${CELER_SOURCE_DIR}" describe --always --abbrev=40 \
    --exclude='*' --dirty) \
    || log warning "no git revision available: set REVISION to label the images"
fi

# Build a stage of the Dockerfile
build() {
  target=$1
  shift
  ${BUILD} \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CONFIG="${CONFIG}" \
    --build-arg GPU_APT_PACKAGES="${GPU_APT_PACKAGES}" \
    --build-arg SPACK_INSTALL_ARGS="${SPACK_INSTALL_ARGS}" \
    --build-arg SPACK_REPO="${SPACK_REPO}" \
    --build-arg SPACK_REF="${SPACK_REF}" \
    --build-arg SPACK_PACKAGES_REPO="${SPACK_PACKAGES_REPO}" \
    --build-arg SPACK_PACKAGES_REF="${SPACK_PACKAGES_REF}" \
    --build-context spack-config="${CELER_SOURCE_DIR}/scripts/spack" \
    --label org.opencontainers.image.created="${CREATED}" \
    --label org.opencontainers.image.revision="${REVISION}" \
    --label org.opencontainers.image.version="${DATE}" \
    --platform linux/amd64 \
    --target "${target}" \
    -t "${target}-${CONFIG}" \
    "$@" \
    "${SCRIPT_DIR}/gpu"
}

build dev "$@"
build ci "$@"

REGISTRY=${REGISTRY:-ghcr.io/celeritas-project}
CI_IMAGE="${REGISTRY}/ci-${CONFIG}:${DATE}"
${CONTAINER_EXE} tag "dev-${CONFIG}" "${REGISTRY}/dev-${CONFIG}:${DATE}"
${CONTAINER_EXE} tag "ci-${CONFIG}" "${CI_IMAGE}"
if [ "${PUSH}" = 1 ]; then
  ${CONTAINER_EXE} push "${CI_IMAGE}"
else
  log info "Built ${CI_IMAGE}: to publish it, run"
  log info "  ${CONTAINER_EXE} push ${CI_IMAGE}"
fi
