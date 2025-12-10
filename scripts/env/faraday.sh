#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! command -v load_system_env >/dev/null 2>&1; then
  printf "error: expected load_system_env helper function via build.sh or shell\n" >&2
  return 1
fi

# From modules/rocmmod
export ROCM_PATH="/opt/rocm-7.0.1"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/lib/llvm/bin:$PATH"
export MANPATH="${ROCM_PATH}/share/man:${ROCM_PATH}/lib/llvm/share/man1:$MANPATH"
export CMAKE_PREFIX_PATH="/opt/rocm-7.0.1:${CMAKE_PREFIX_PATH}"
export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?
