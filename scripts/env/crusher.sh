#!/bin/sh -e

_celer_base=$PROJWORK/csc404/celeritas-crusher

# From %clang@14.0.0-rocm5.1.0 in OLCF's spack compilers:
module load PrgEnv-amd/8.3.3 amd/5.1.0 craype-x86-trento libfabric \
  cray-pmi/6.1.2 cray-python/3.9.13.1
export RFE_811452_DISABLE=1
export LD_LIBRARY_PATH=/opt/cray/pe/pmi/6.1.2/lib:$LD_LIBRARY_PATH:/opt/cray/pe/gcc-libs:/opt/cray/libfabric/1.15.0.0/lib64
export LIBRARY_PATH=/opt/rocm-5.1.0/lib:/opt/rocm-5.1.0/lib64:$LIBRARY_PATH

# Avoid linking multiple different libsci (one with openmp, one without)
module unload cray-libsci

# Set up compilers
test -n "${CRAYPE_DIR}"
export CXX=${CRAYPE_DIR}/bin/CC
export CC=${CRAYPE_DIR}/bin/cc

# Do NOT load the accelerator target, because it adds
# -fopenmp-targets=amdgcn-amd-amdhsa 
# -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
# which implicitly defines __CUDA_ARCH__
# module load craype-accel-amd-gfx90a

# Set up celeritas
export SPACK_ROOT=/ccs/proj/csc404/spack-crusher
export PATH=${_celer_base}/spack/view/bin:$PATH
export CMAKE_PREFIX_PATH=${_celer_base}/spack/view:${CMAKE_PREFIX_PATH}
export MODULEPATH=${SPACK_ROOT}/share/spack/lmod/cray-sles15-x86_64/Core:${MODULEPATH}
module load geant4-data

# Make llvm available
test -n "${ROCM_PATH}"
export PATH=$PATH:${ROCM_PATH}/llvm/bin
