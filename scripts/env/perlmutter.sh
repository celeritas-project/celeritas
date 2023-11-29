#!/bin/sh -e

# The cudatoolkit on Perlmutter is provided by NVHPC, which moved some CUDA headers to a "math_libs" directory (e.g., curand_kernel.h).
# If using Spack, this will cause a compile error in VecGeom+cuda.
# Spack uses this CUDA install as external CUDA package and doesn't look in the math_libs directory for extra headers.
# --> the fix for now is to unload these modules and install our own cudatoolkit using Spack.
module unload gpu cudatoolkit
module load PrgEnv-gnu/8.3.3

# Spack module on Perlmutter currently fails to create the spack env from spack.yaml, we need Spack v0.18.0; use our own install instead.
# Expects the spack git repo to have been cloned at _SPACK_INSTALL and the environment celeritas to exist
_SPACK_INSTALL=${SCRATCH}/spack
_SPACK_SOURCE_FILE=${_SPACK_INSTALL}/share/spack/setup-env.sh
if [ ! -f "${_SPACK_SOURCE_FILE}" ]; then
    echo "Expected to find a spack install at ${_SPACK_INSTALL}" >&2
    exit 2
fi

. ${_SPACK_SOURCE_FILE}
spack env activate celeritas
export LD_LIBRARY_PATH=$SPACK_ENV/.spack-env/view/lib64:$LD_LIBRARY_PATH