#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if ! command -v load_system_env >/dev/null 2>&1; then
  printf "error: define a function load_system_env in your shell rc:
load_system_env() {
  . \${CELER_SOURCE_DIR}/scripts/env/\$1.sh
}
" >&2
  return 1
fi

# FIXME: scratch isn't mounted on faraday :(
export SCRATCHDIR=/tmp/${USER}

# From modules/rocmmod
export PATH=/opt/rocm-7.0.1/bin:/opt/rocm-7.0.1/lib/llvm/bin:$PATH
export MANPATH="/opt/rocm-7.0.1/share/man:/opt/rocm-7.0.1/lib/llvm/share/man1:$MANPATH"
export CMAKE_PREFIX_PATH="/opt/rocm-7.0.1:$CMAKE_PREFIX_PATH"
export ROCM_PATH="/opt/rocm-7.0.1"
export HIP_PATH="/opt/rocm-7.0.1"

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?
