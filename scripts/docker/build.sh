#!/bin/sh -ex
# Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "$1" ]; then
  echo "usage: $0 config"
  exit 2
fi

SPACK_VERSION=v0.19.0
CONFIG=$1

case $CONFIG in 
  minimal)
    CONFIG=bionic-minimal
    ;;
  cuda)
    CONFIG=focal-cuda11
    ;;
  hip)
    CONFIG=centos7-rocm5
    ;;
esac
 
DOCKERFILE_DISTRO=ubuntu
case $CONFIG in 
  bionic-minimal)
    DOCKERFILE_DISTRO=ubuntu
    BASE_TAG=ubuntu:bionic-20210930
    VECGEOM=
    ;;
  focal-cuda11)
    # ***IMPORTANT***: update cuda external version in dev/focal-cuda11!
    DOCKERFILE_DISTRO=ubuntu
    BASE_TAG=nvidia/cuda:11.4.2-devel-ubuntu20.04
    VECGEOM=v1.1.18
    ;;
  centos7-rocm5)
    # ***IMPORTANT***: update hip external version in dev/centos7-rocm5!
    BASE_TAG=rocm/dev-centos-7:5.4
    VECGEOM=
    DOCKERFILE_DISTRO=centos
    ;;
  *)
    echo "Invalid configure type: $1"
    exit 1
    ;;
esac

docker pull ${BASE_TAG}
docker tag ${BASE_TAG} base-${CONFIG}

docker build -t dev-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg SPACK_VERSION=${SPACK_VERSION} \
  --build-arg DOCKERFILE_DISTRO=${DOCKERFILE_DISTRO} \
  dev

docker build -t ci-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg VECGEOM=${VECGEOM} \
  ci

DATE=$(date '+%Y-%m-%d')
docker tag dev-${CONFIG} celeritas/dev-${CONFIG}:${DATE}
docker tag ci-${CONFIG} celeritas/ci-${CONFIG}:${DATE}
docker push celeritas/ci-${CONFIG}:${DATE}
