#!/bin/sh -e

# The cudatoolkit on Perlmutter is provided by NVHPC, which moved some CUDA headers to a "math_libs" directory (e.g., curand_kernel.h).
# If using Spack, this will cause a compile error in VecGeom+cuda.
# Spack uses this CUDA install as external CUDA package and doesn't look in the math_libs directory for extra headers.
# --> the fix for now is to unload these modules and install our own cudatoolkit using Spack.
module unload gpu cudatoolkit
module load PrgEnv-gnu/8.3.3

# Spack module on Perlmutter currently fails to create the spack env from spack.yaml, so we use our own install
_SPACK_INSTALL=${SCRATCH}/spack

. ${_SPACK_INSTALL}/share/spack/setup-env.sh
spack env activate celeritas
