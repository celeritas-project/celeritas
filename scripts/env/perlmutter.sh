#!/bin/sh -e

# Expects the spack git repo to have been cloned at _SPACK_INSTALL (default to $SPACK_ROOT)
# The environment named celeritas must exists
_SPACK_INSTALL=${SPACK_ROOT:-$SCRATCH/spack}
_SPACK_SOURCE_FILE=${_SPACK_INSTALL}/share/spack/setup-env.sh
if [ ! -f "${_SPACK_SOURCE_FILE}" ]; then
    echo "Expected to find a spack install at ${_SPACK_INSTALL}" >&2
    exit 2
fi

. ${_SPACK_SOURCE_FILE}
spack env activate celeritas
