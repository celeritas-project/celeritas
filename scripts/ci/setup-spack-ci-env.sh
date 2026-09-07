#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set -e

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$(dirname $0)"/../.. && pwd)
fi
export CELER_SOURCE_DIR # Used by spack environment script

SPACK=$(command -v spack 2>/dev/null || printf "")

if [ -z "$SPACK" ]; then
  echo "error: spack not found"
  exit 1
fi

if [ -n "$SPACK_ENV" ]; then
  echo "error: another spack environment is active: run despacktivate first"
  exit 1
fi

# Configure separate packages repository *first*: otherwise the
# environment creation will do a lengthy unnecessary checkout
if [ -n "${SPACK_PACKAGES}" ]; then
  echo "Using custom builtin spack package repo: ${SPACK_PACKAGES}"
  $SPACK repo set --destination "${SPACK_PACKAGES}" builtin
else
  SPACK_PACKAGES=$(spack location -P builtin)
  echo "Using default builtin spack repo: ${SPACK_PACKAGES}"
fi

# Create environment in current working directory
echo "Creating environment"
$SPACK env create . "${CELER_SOURCE_DIR}/scripts/spack/env-ci-base.yaml"

# Configure install prefix
if [ -n "${CELER_SPACK_OPT}" ]; then
  echo "Setting spack install prefix to ${CELER_SPACK_OPT}"
  $SPACK -e . config add "config:install_tree:root:${CELER_SPACK_OPT}"
else
  echo "Omitting spack install prefix: CELER_SPACK_OPT is not set"
fi

# Configure view
if [ -n "${CELER_SPACK_VIEW}" ]; then
  echo "Setting spack view to ${CELER_SPACK_VIEW}"
else
  echo "Omitting spack view: CELER_SPACK_VIEW is not set"
  CELER_SPACK_VIEW=false
fi
$SPACK -e . config add "view:${CELER_SPACK_VIEW}"

# Configure C++ standard
if [ -n "${CXXSTD}" ]; then
  echo "Setting cxxstd preference to ${CXXSTD}"
  $SPACK -e . config add "packages:all:prefer:[cxxstd=${CXXSTD}]"
else
  echo "warning: no cxxstd preference: CXXSTD not set"
fi

if [ $# -ne 0 ] ; then
  echo "Adding additional packages: $@"
  $SPACK -e . add "$@"
fi

# Add the spack ref so that updating spack will reconcretize
cat >> spack.yaml <<EOF
# spack: $(git -C "${SPACK_ROOT}" log -1 --pretty=%H HEAD)
# packages: $(git -C "${SPACK_PACKAGES}" log -1 --pretty=%H HEAD)
EOF
