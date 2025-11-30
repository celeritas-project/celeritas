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

# Dispatch common loading to the 'excl' system
load_system_env excl || return $?

# Load AMD modules
if command -v module >/dev/null 2>&1; then
  module load rocmmod
else
  celerlog error "module: command not found"
  return 1
fi
