#!/bin/sh
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Called via scripts/ci/update-spack-buildcache.sh for main use
# OR run directly to print the configurations
# OR define build_spack_env and `eval` the output
#
# Each line is:
#   CXXSTD=<STD> ENV_CI=<WHAT> build_spack_env <additional spack packages>
# where <WHAT> selects scripts/spack/env-ci-<WHAT>.yaml.
#
# This should correspond to the matrix (and its "include" entries) from
# .github/workflows/build-spack.yml and other spack-based builds
# (missing concretization will require running the build-spack workflows
# on something *other* than a PR, and missing packages will cause a PR to fail
# due to the `--use-buildcache` option in `setup-spack/action.yaml`)
#-----------------------------------------------------------------------------#

if ! command build_spack_env >/dev/null 2>&1; then
  echo "# Dry run: launch using scripts/ci/update-spack-buildcache.sh"
  build_spack_env() {
    echo "CXXSTD=${CXXSTD} ENV_CI=${ENV_CI} build_spack_env $*"
  }
fi

# build-spack
for G4_VERSION in 11.3 11.2 11.1 11.0 10.7; do
  CXXSTD=20 ENV_CI=base build_spack_env vecgeom@2.2.0 geant4@${G4_VERSION} g4vg root
done
for G4_VERSION in 10.6 10.5; do
  CXXSTD=17 ENV_CI=base build_spack_env vecgeom@2.2.0 geant4@${G4_VERSION} g4vg hepmc3~rootio
done
CXXSTD=20 ENV_CI=base build_spack_env vecgeom@1.2.11 geant4@11.3 g4vg root

# UNUSED (will add to CI later)
CXXSTD=20 ENV_CI=base build_spack_env vecgeom@2.2.0 geant4@11.4 g4vg root dd4hep

# build-codecov.spack
CXXSTD=20 ENV_CI=base build_spack_env vecgeom@2.2.0 geant4@11.3 g4vg root py-gcovr
CXXSTD=20 ENV_CI=base build_spack_env vecgeom@2.2.0 geant4@11.3+vecgeom

# build-spack-extra.ancient
CXXSTD=17 ENV_CI=ancient build_spack_env
