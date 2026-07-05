#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! command -v load_system_env >/dev/null 2>&1; then
  printf "error: expected load_system_env helper function via build.sh or shell\n" >&2
  return 1
fi

export CXX=/usr/bin/g++-13
export CC=/usr/bin/gcc-13

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?

export CUDAARCHS=70
export CUDAFLAGS="-Werror all-warnings -Wno-deprecated-gpu-targets"
export CUDA_HOME=${CELERITAS_OPT}/cuda/12.9.1/pmicvvf
export CUDACXX=${CUDA_HOME}/bin/nvcc
