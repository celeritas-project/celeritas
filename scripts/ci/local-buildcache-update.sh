#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run on excl (or any ubuntu24 system) to build and upload
#-----------------------------------------------------------------------------#

set -e

if [ -z "${GITHUB_USER}" || -z "${GITHUB_TOKEN}" ]; then
  echo "error: GITHUB_USER and GITHUB_TOKEN must be set (see scripts/spack/reqs-ci.yaml)"
  exit 1
fi

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

update_index=false

# Each line is: CXXSTD, followed by the spack packages to add, based on the
# matrix (and its "include" entries) from .github/workflows/build-spack.yml
matrix="
CXXSTD=20 vecgeom@2.1.0 geant4@11.4 g4vg root
CXXSTD=20 vecgeom@2.0.0-rc.7 geant4@11.3 g4vg root
CXXSTD=20 vecgeom@1.2.11 geant4@11.4 g4vg root py-gcovr
CXXSTD=20 vecgeom@1.2.11 geant4@11.3 g4vg root
CXXSTD=20 vecgeom@1.2.11 geant4@11.2 g4vg root
CXXSTD=20 vecgeom@1.2.11 geant4@11.1 g4vg root
CXXSTD=20 vecgeom@1.2.11 geant4@11.0 g4vg root
CXXSTD=20 vecgeom@1.2.11 geant4@10.7 g4vg root
CXXSTD=17 vecgeom@1.2.11 geant4@10.6 g4vg
CXXSTD=17 vecgeom@1.2.11 geant4@10.5 g4vg
"

echo "$matrix" | while read -r line; do
  [ -z "$line" ] && continue
  # Pop and export CXXSTD
  set -- $line
  export "$1"
  shift

  # Create temporary directory
  envdir="$WORK_DIR/temp-spack-cxx${CXXSTD}-$(echo "$*" | tr ' @' '--')"
  if [ -d $envdir ]; then
    echo "Skipping existing env: $line"
    continue
  fi
  mkdir -p "$envdir"
  cd "$envdir"

  # Create environment
  "${SCRIPT_DIR}/setup-spack-ci-env.sh" "$@"
  # Install and push
  echo "Concretizing $envdir..."
  spack -e . -v concretize --non-defaults --fresh
  echo "Installing  $envdir..."
  spack -e . install
  echo "Pushing  $envdir..."
  spack -e . buildcache push \
    --base-image $CELER_BASE_IMAGE \
    $CELER_BUILDCACHE
  update_index=true
done

if $update_index; then
  # Should be inside a valid environment
  spack -e . buildcache update-index $CELER_BUILDCACHE
fi
