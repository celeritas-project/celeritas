#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Build the GPU development and CI images with the Spack revisions pinned by
# .github/actions/setup-spack and tag them with the date. The CI image is
# pushed only if PUSH=1 is set. SPACK_INSTALL_ARGS (e.g. "-p 8 -j 32") is
# passed to 'spack install'.
# Additional arguments are passed to each build command, e.g. --no-cache.
# See scripts/docker/_container.sh for the container engine selection.
#-----------------------------------------------------------------------------#

if [ -z "$1" ]; then
  echo "usage: $0 {cuda|rocm|ubuntu-cuda13|ubuntu-rocm7} [build args...]" >&2
  exit 2
fi
CONFIG=$1
shift

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CELER_SOURCE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

. "${SCRIPT_DIR}/_container.sh"

case ${CONFIG} in
  cuda )
    CONFIG=ubuntu-cuda13
    ;;
  rocm )
    CONFIG=ubuntu-rocm7
    ;;
esac

# Use the newest toolkit versions known to the pinned spack-packages.
# When updating a base image, also update the versions listed in README.md
# and the image tags in .github/workflows/build-{docker,codecov}.yml
case ${CONFIG} in
  ubuntu-cuda13)
    BASE_IMAGE=nvidia/cuda:13.4.1-devel-ubuntu24.04
    ;;
  ubuntu-rocm7)
    # The "complete" image provides the rocrand/rocthrust/hipcub libraries
    BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.3-complete
    ;;
  *)
    log error "invalid configuration: ${CONFIG}"
    exit 1
    ;;
esac

# Build a stage of the Dockerfile
build() {
  target=$1
  shift
  ${BUILD} \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg CONFIG="${CONFIG}" \
    --build-arg SPACK_INSTALL_ARGS="${SPACK_INSTALL_ARGS}" \
    --build-arg SPACK_REPO="${SPACK_REPO}" \
    --build-arg SPACK_REF="${SPACK_REF}" \
    --build-arg SPACK_PACKAGES_REPO="${SPACK_PACKAGES_REPO}" \
    --build-arg SPACK_PACKAGES_REF="${SPACK_PACKAGES_REF}" \
    --build-context spack-config="${CELER_SOURCE_DIR}/scripts/spack" \
    --platform linux/amd64 \
    --target "${target}" \
    -t "${target}-${CONFIG}" \
    "$@" \
    "${SCRIPT_DIR}/gpu"
}

build dev "$@"
build ci "$@"

DATE=$(date '+%Y-%m-%d')
${CONTAINER_EXE} tag "dev-${CONFIG}" "celeritas/dev-${CONFIG}:${DATE}"
${CONTAINER_EXE} tag "ci-${CONFIG}" "celeritas/ci-${CONFIG}:${DATE}"
if [ "${PUSH}" = 1 ]; then
  ${CONTAINER_EXE} push "celeritas/ci-${CONFIG}:${DATE}"
else
  log info "Built celeritas/ci-${CONFIG}:${DATE}: to publish it, run"
  log info "  ${CONTAINER_EXE} push celeritas/ci-${CONFIG}:${DATE}"
fi
