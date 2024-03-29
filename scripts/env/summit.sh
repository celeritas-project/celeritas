#!/bin/sh -e

_celer_base=$PROJWORK/csc404/celeritas
export LLVM_ROOT=/sw/summit/llvm/14.0.0/14.0.0-0

module load cuda/11.4.2 gcc/11.2 spectrum-mpi/10.4
module unload cmake python
export PATH=${_celer_base}/spack/view/bin:$LLVM_ROOT/bin:$PATH
export CMAKE_PREFIX_PATH=${_celer_base}/spack/view:${CMAKE_PREFIX_PATH}
export MODULEPATH=/ccs/proj/csc404/spack/share/spack/lmod/linux-rhel8-ppc64le/Core:$MODULEPATH
export CXX=${OLCF_GCC_ROOT}/bin/g++
export CC=${OLCF_GCC_ROOT}/bin/gcc
export CUDACXX=${OLCF_CUDA_ROOT}/bin/nvcc
module load geant4-data
