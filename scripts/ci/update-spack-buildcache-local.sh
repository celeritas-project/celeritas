#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run on excl (or any ubuntu24 system) to build and upload
#-----------------------------------------------------------------------------#

set -e
log() {
  printf "%s: %s\n" "$1" "$2" >&2
}

if [ -z "${GITHUB_USER}" ] || [ -z "${GITHUB_TOKEN}" ]; then
  log error "GITHUB_USER and GITHUB_TOKEN must be set (see scripts/spack/reqs-ci.yaml)"
  exit 1
fi

if ! command -v spack 2>/dev/null; then
  log error "spack not found"
  exit 1
fi

OS=$(spack arch --operating-system)
EXPECTED_OS="ubuntu24.04"
if [ "${OS}" != "${EXPECTED_OS}" ]; then
  log error "Current OS is ${OS} but needs to be ${EXPECTED_OS}"
  exit 1
fi

spack debug report

CELER_BASE_IMAGE=ubuntu:24.04
CELER_BUILDCACHE=celeritas

export CELER_SPACK_VIEW=false
CELER_SPACK_OPT=/scratch/celeritas/opt
if [ -d "$CELER_SPACK_OPT" ]; then
  export CELER_SPACK_OPT
fi

WORK_DIR=$PWD
SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
export CELER_SOURCE_DIR=$(cd $SCRIPT_DIR/../.. && pwd)

# Each line is: CXXSTD, env-ci-{WHAT}.yaml, additional spack packages to add
# This should correspond to the matrix (and its "include" entries) from
# .github/workflows/build-spack.yml and other spack-based builds
# (missing concretization will require running the build-spack workflows
# on something *other* than a PR, and missing packages will cause a PR to fail
# due to the `--use-buildcache` option in `setup-spack/action.yaml`)
matrix="
20 base vecgeom@2.1.0 geant4@11.4 g4vg root dd4hep
20 base vecgeom@2.0.0-rc.7 geant4@11.3 g4vg root
20 base vecgeom@1.2.11 geant4@11.4 g4vg root py-gcovr
20 base vecgeom@1.2.11 geant4@11.3 g4vg root
20 base vecgeom@1.2.11 geant4@11.2 g4vg root
20 base vecgeom@1.2.11 geant4@11.1 g4vg root
20 base vecgeom@1.2.11 geant4@11.0 g4vg root
20 base vecgeom@1.2.11 geant4@10.7 g4vg root
17 base vecgeom@1.2.11 geant4@10.6 g4vg
17 base vecgeom@1.2.11 geant4@10.5 g4vg
17 ancient
"

printf "%s" "$matrix" | while read -r line; do
  [ -z "$line" ] && continue
  # Convert line into arguments
  set -- $line
  cxxstd=$1
  envbase=$2
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
    "${SCRIPT_DIR}/setup-spack-ci-env.sh" "$@"
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
