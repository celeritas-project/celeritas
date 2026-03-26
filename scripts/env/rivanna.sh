#!/bin/sh -e

# The A100 GPU nodes in Rivanna/Afton have a different architecture from the rest of the nodes on the cluster.
# Ensure the spack environment is built with the target x86_64_v3 for cross-platform support.

# Similar with Perlmutter, do not use NVHPC but instead GCC and CUDA modules directly
# Miniforge is load for python support on different nodes
module load miniforge/24.11.3-py3.12 gcc/14.2.0 cuda/13.0.2

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
