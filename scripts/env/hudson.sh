#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

module load nvhpc-byo-compiler/25.7

export SPACK_ROOT=/auto/projects/celeritas/spack
export CXX=/usr/bin/c++
export CUDAHOSTCXX=$CXX

if ! command -v spack 2>/dev/null; then
  . $SPACK_ROOT/share/spack/setup-env.sh
fi

for _d in build install; do
  # Create build/install in higher-performance local-but-persistent dir
  _scratch="/scratch/$USER/$_d"
  if ! test -d $_scratch; then
    mkdir -p $_scratch
    chmod 700 $_scratch
  fi
done

CELERITAS_ENV=${SPACK_ROOT}/var/spack/environments/celeritas/.spack-env/view
export PATH=${CELERITAS_ENV}/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELERITAS_ENV}:${CMAKE_PREFIX_PATH}
