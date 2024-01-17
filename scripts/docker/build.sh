#!/bin/sh -ex
# Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "$1" ]; then
  echo "usage: $0 config"
  exit 2
fi

SPACK_VERSION=v0.19.0
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
  minimal )
    CONFIG=bionic-minimal
    ;;
  cuda )
    # When updating: change here, dev/{name}.yaml, dev/launch-local-test.sh
    CONFIG=jammy-cuda11
    ;;
  hip )
    CONFIG=centos7-rocm5
    ;;
  rocm )
    CONFIG=centos7-rocm5
    ;;
esac
 
case $CONFIG in 
  bionic-minimal)
    DOCKERFILE_DISTRO=ubuntu
    BASE_TAG=ubuntu:bionic-20221019
    VECGEOM=
    ;;
  jammy-cuda11)
    # ***IMPORTANT***: update cuda external version in dev/jammy-cuda11!
    DOCKERFILE_DISTRO=ubuntu
    BASE_TAG=nvidia/cuda:11.8.0-devel-ubuntu22.04
    VECGEOM=v1.2.5
    ;;
  centos7-rocm5)
    # ***IMPORTANT***: update hip external version in dev/centos7-rocm5!
    DOCKERFILE_DISTRO=centos
    BASE_TAG=rocm/dev-centos-7:5.4
    VECGEOM=
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
  --build-arg VECGEOM=${VECGEOM} \
  --build-arg DOCKERFILE_DISTRO=${DOCKERFILE_DISTRO} \
  ${BUILDARGS} \
  ci

DATE=$(date '+%Y-%m-%d')
${DOCKER} tag dev-${CONFIG} celeritas/dev-${CONFIG}:${DATE}
${DOCKER} tag ci-${CONFIG} celeritas/ci-${CONFIG}:${DATE}
${DOCKER} push celeritas/ci-${CONFIG}:${DATE}
