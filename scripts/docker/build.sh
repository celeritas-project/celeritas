#!/bin/sh -ex
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "$1" ]; then
  echo "usage: $0 config"
  exit 2
fi

SPACK_VERSION=v0.16.0
CONFIG=$1

case $CONFIG in 
  minimal)
    CONFIG=bionic-minimal
    ;;
  cuda)
    CONFIG=focal-cuda11
    ;;
esac
 
case $CONFIG in 
  bionic-minimal)
    BASE_TAG=ubuntu:bionic-20210428
    VECGEOM=
    ;;
  focal-cuda11)
    BASE_TAG=nvidia/cuda:11.1-devel-ubuntu20.04
    VECGEOM=v1.1.13
    ;;
  *)
    echo "Invalid configure type: $1"
    exit 1
    ;;
esac

docker tag ${BASE_TAG} base-${CONFIG}

docker build -t dev-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg SPACK_VERSION=${SPACK_VERSION} \
  dev

docker build -t ci-${CONFIG} \
  --build-arg CONFIG=${CONFIG} \
  --build-arg VECGEOM=${VECGEOM} \
  ci

DATE=$(date '+%Y-%m-%d')
docker tag dev-${CONFIG} celeritas/dev-${CONFIG}:${DATE}
docker tag ci-${CONFIG} celeritas/ci-${CONFIG}:${DATE}
docker push celeritas/ci-${CONFIG}:${DATE}
