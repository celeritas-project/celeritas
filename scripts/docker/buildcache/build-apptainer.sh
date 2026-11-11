#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run on excl to build and push apptainer image
#-----------------------------------------------------------------------------#


if [ -z "$SCRATCHDIR" ]; then
  SCRATCHDIR="$SCRATCH"
  if [ -z "$SCRATCHDIR" ]; then
    log error "no scratch directory defined"
    exit 1
  fi
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

. "${SCRIPT_DIR}/build.sh" "$@"
# Check output variables are defined
test -n "$CONTAINER_EXE" && test -n "$PROJECT" && test -n "$TAG"

if ! have apptainer; then
  log error "Expected apptainer"
fi


if [ "$CONTAINER_EXE" = podman ] || [ "$CONTAINER_EXE" = podman-hpc ]; then
  OUTPUT_ARGS=--format docker-archive -o
else
  OUTPUT_ARGS=-o
fi
OUTPUT_DIR=$SCRATCHDIR/celeritas/containers
REPO_HOST=camden.ornl.gov

convert() {
  target=$1
  shift
  ARCHIVE="${OUTPUT_DIR}/${target}-${TAG}.tar"
  SIF="${OUTPUT_DIR}/${target}-${TAG}.sif"

  if ! [ -e "${ARCHIVE}" ]; then
    log info "Exporting image"
    "${CONTAINER_EXE}" save ${OUTPUT_ARGS} "${ARCHIVE}" "${PROJECT}/${target}:${TAG}"
  else
    log info "Skipping image export: $ARCHIVE exists"
  fi

  if ! [ -e "${SIF}" ]; then
    log info "Building with apptainer"
    apptainer build "${SIF}" "docker-archive://${ARCHIVE}"
  else
    log info "Skipping apptainer export: $SIF exists"
  fi
}

push() {
  target=$1
  shift
  SIF="${OUTPUT_DIR}/${target}-${TAG}.sif"
  apptainer push "${SIF}" "oras://${REPO_HOST}/${PROJECT}/${target}:${TAG}"
}

convert spack-base-ubuntu24
convert buildcache
push spack-base-ubuntu24
push buildcache
