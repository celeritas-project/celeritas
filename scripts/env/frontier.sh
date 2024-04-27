#!/bin/sh -e

PROJID=hep143
_celer_view=${PROJWORK}/${PROJID}/opt-view
_tool_view=/ccs/proj/${PROJID}/opt-view
_conda=/ccs/proj/${PROJID}/conda-frontier

module load PrgEnv-amd/8.5.0 cpe/23.12 amd/5.7.1 craype-x86-trento \
  libfabric/1.15.2.0 miniforge3/23.11.0
# Disable warning "Using generic mem* routines instead of tuned routines"
export RFE_811452_DISABLE=1

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
export SPACK_ROOT=/ccs/proj/hep143/spack
export PATH=${_celer_view}/bin:${_tool_view}/bin:${_conda}/bin:$PATH
export CMAKE_PREFIX_PATH=${_celer_view}:${CMAKE_PREFIX_PATH}
export MODULEPATH=${PROJWORK}/${PROJID}/share/lmod/linux-sles15-x86_64/Core:${MODULEPATH}

# Set up Geant4 data 
module load geant4-data/11.0
test -n "${G4ENSDFSTATEDATA}"
test -e "${G4ENSDFSTATEDATA}"

# Make llvm available
test -n "${ROCM_PATH}"
export PATH=$PATH:${ROCM_PATH}/llvm/bin
