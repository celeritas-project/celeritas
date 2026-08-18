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

export ROCM_PATH="/opt/rocm/core-7.14"
export HIP_PATH="${ROCM_PATH}"
export PATH="${ROCM_PATH}/lib/llvm/bin:$PATH"
export MANPATH="${ROCM_PATH}/share/man:${ROCM_PATH}/lib/llvm/share/man1:$MANPATH"
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${CMAKE_PREFIX_PATH}"

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?
