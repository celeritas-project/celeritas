#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! command -v load_system_env >/dev/null 2>&1; then
  printf "error: expected load_system_env helper function via build.sh or shell\n" >&2
  return 1
fi

# Redundant with cmake prefix but useful if this is being used for other env
export CUDAARCHS=70
# Set C++ compiler
export CXX=/usr/bin/c++
export CC=/usr/bin/cc

export CUDAFLAGS="-Werror all-warnings -Wno-deprecated-gpu-targets"

export CUDA_HOME=/auto/projects/celeritas/opt/ubuntu24.04/x86_64_v3/cuda/12.9.1/nnvx55h
export CUDACXX=${CUDA_HOME}/bin/nvcc

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?
