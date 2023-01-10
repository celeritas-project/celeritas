#!/bin/sh -ex
# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

if [ -z "${CELER_INSTALL_DIR}" ]; then
  # Assume run-local.sh layout
  CELER_INSTALL_DIR="$(git rev-parse --show-toplevel)/install"
fi

CI_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "${CI_SCRIPT_DIR}/test-installation"
git clean -fxd .
mkdir build
cd build
export CMAKE_PREFIX_PATH=${CELER_INSTALL_DIR}:${CMAKE_PREFIX_PATH}
cmake -G Ninja \
  -D CMAKE_INSTALL_PREFIX=${CI_SCRIPT_DIR}/install \
  ..
ninja
exec ./example
