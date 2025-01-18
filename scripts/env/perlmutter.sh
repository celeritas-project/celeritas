#!/bin/sh -e

# The cudatoolkit on Perlmutter is provided by NVHPC, which moved some CUDA headers to a "math_libs" directory (e.g., curand_kernel.h).
# If using Spack, this will cause a compile error in VecGeom+cuda.
# Spack uses this CUDA install as external CUDA package and doesn't look in the math_libs directory for extra headers.
# --> the fix for now is to unload these modules and install our own cudatoolkit using Spack.
module unload gpu
module load PrgEnv-gnu
module unload cray-libsci

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

export PKG_CONFIG_PATH=/opt/cray/xpmem/default/lib64/pkgconfig:"${PKG_CONFIG_PATH}"
