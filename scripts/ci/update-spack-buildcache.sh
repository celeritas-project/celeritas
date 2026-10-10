#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run on excl (or any ubuntu24 system) to build and upload, or use the
# container in scripts/docker/buildcache (see scripts/docker/README.md)
#-----------------------------------------------------------------------------#

set -e

if ! command celerlog >/dev/null 2>&1; then
  celerlog() {
    printf "%s: %s\n" "$1" "$2" >&2
  }
fi
log() {
  celerlog "$@"
}

if [ -z "${GITHUB_USER}" ] || [ -z "${GITHUB_TOKEN}" ]; then
  log error "GITHUB_USER and GITHUB_TOKEN must be set (see scripts/spack/reqs-ci.yaml)"
  exit 1
fi

if ! command -v spack >/dev/null 2>&1; then
  log error "spack not found"
  exit 1
fi

OS=$(spack arch --operating-system)
EXPECTED_OS="ubuntu24.04"
if [ "${OS}" != "${EXPECTED_OS}" ]; then
  log error "Current OS is ${OS} but needs to be ${EXPECTED_OS}"
  exit 1
fi

# Report git sha of spack repo and builtin:
# these should be configured before running the script
# using parse-spack-versions.py
spack debug report

CELER_BASE_IMAGE=ubuntu:24.04
CELER_BUILDCACHE=celeritas

export CELER_SPACK_VIEW=false
CELER_SPACK_OPT=/scratch/celeritas/opt-ci
if [ -d "$CELER_SPACK_OPT" ]; then
  export CELER_SPACK_OPT
else
  log warning "CELER_SPACK_OPT=${CELER_SPACK_OPT} does not exist"
fi

WORK_DIR=$PWD
if [ -z "${CELER_SOURCE_DIR}" ]; then
  SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
  CELER_SOURCE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
fi

MATRIX_FILE="${CELER_SOURCE_DIR}/.github/buildcache-matrix.txt"
if [ ! -f "$MATRIX_FILE" ]; then
  log error "Missing matrix file ${MATRIX_FILE}"
  exit 1
fi

grep -v '^[[:space:]]*\(#\|$\)' "$MATRIX_FILE" | while read -r line; do
  # Convert line into arguments to be passed into setup-spack-ci-env.sh
  set -- $line
  case "$1:$2" in
    cxxstd=*:envbase=*) ;;
    *)
      log error "Invalid matrix line (expected cxxstd=N envbase=NAME): $line"
      exit 1
      ;;
  esac
  cxxstd=${1#cxxstd=}
  envbase=${2#envbase=}
  shift 2

  # Create temporary directory
  envdir="$WORK_DIR/temp-spack-${envbase}-cxx${cxxstd}"
  if [ $# -ne 0 ]; then
    envdir="${envdir}-$(echo "$*" | tr ' @' '--')"
  fi
  if [ -d $envdir ]; then
    log info "Skipping existing env: $line"
    continue
  fi
  mkdir -p "$envdir"
  cd "$envdir"

  # Create environment
  SPACK_ENV_FILE="env-ci-${envbase}.yaml" \
  CXXSTD=${cxxstd} \
  CELER_SOURCE_DIR=${CELER_SOURCE_DIR} \
    "${CELER_SOURCE_DIR}/scripts/ci/setup-spack-ci-env.sh" "$@"
  # Install and push
  log status "Concretizing $envdir"
  spack -e . -v concretize --non-defaults --fresh
  log status "Installing $envdir"
  spack -e . install
  log status "Pushing $envdir"
  spack -e . buildcache push \
    --base-image $CELER_BASE_IMAGE \
    --update-index \
    --allow-missing \
    $CELER_BUILDCACHE
done
