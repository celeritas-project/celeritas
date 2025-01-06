#!/bin/sh -ex
# Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "$1" ]; then
  echo "usage: $0 config"
  exit 2
fi

# Note: when changing spack version, UPDATE PATCHES in dev/Dockerfile
SPACK_VERSION=develop-2024-12-22
CONFIG=$1
DOCKER=docker
BUILDARGS=
if ! hash ${DOCKER} 2>/dev/null; then
  # see https://blog.christophersmart.com/2021/01/26/user-ids-and-rootless-containers-with-podman/
  DOCKER=podman
  BUILDARGS="--format docker"
  if ! hash ${DOCKER} 2>/dev/null; then
    echo "Docker (or podman) is not available"
    exit 1
  fi
  # Make podman build containers inside /tmp rather than /var/lib
  export TMPDIR=$(mktemp -d)
fi

case $CONFIG in 
  cuda )
    # When updating: change here, dev/{name}.yaml, dev/launch-local-test.sh
    CONFIG=rocky-cuda12
    ;;
  rocm )
    CONFIG=ubuntu-rocm6
    ;;
esac
 
case $CONFIG in
  rocky-cuda12)
    # ***IMPORTANT***: update the following after modification
    # - cuda external version in dev/rocky-cuda12
    # - CI versions listed in README.md
    DOCKERFILE_DISTRO=rocky
    BASE_TAG=nvidia/cuda:12.6.3-devel-rockylinux9
    ;;
  ubuntu-rocm6)
    # ***IMPORTANT***: update hip external version in dev/ubuntu-rocm5!
    DOCKERFILE_DISTRO=ubuntu
    BASE_TAG=rocm/dev-ubuntu-24.04:6.3.1
    ;;
  *)
    echo "Invalid configure type: $1"
    exit 1
    ;;
esac

${DOCKER} pull ${BASE_TAG}
${DOCKER} tag ${BASE_TAG} base-${CONFIG}

${DOCKER} build -t dev-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg SPACK_VERSION=${SPACK_VERSION} \
  --build-arg DOCKERFILE_DISTRO=${DOCKERFILE_DISTRO} \
  ${BUILDARGS} \
  dev

${DOCKER} build -t ci-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg DOCKERFILE_DISTRO=${DOCKERFILE_DISTRO} \
  ${BUILDARGS} \
  ci

DATE=$(date '+%Y-%m-%d')
${DOCKER} tag dev-${CONFIG} celeritas/dev-${CONFIG}:${DATE}
${DOCKER} tag ci-${CONFIG} celeritas/ci-${CONFIG}:${DATE}
${DOCKER} push celeritas/ci-${CONFIG}:${DATE}
