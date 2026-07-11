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
export CUDAARCHS=120
# Set C++ compiler
export CXX=/usr/bin/g++-13
export CC=/usr/bin/gcc-13

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?
