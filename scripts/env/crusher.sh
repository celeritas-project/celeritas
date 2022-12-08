#!/bin/sh -e

module load DefApps/default PrgEnv-gnu craype-accel-amd-gfx90a rocm
module load nlohmann-json cmake ninja googletest
module try_load root clhep xerces-c expat geant4 hepmc3
module load cray-python/3.9.13.1
module unload cray-libsci

# NOTE: these might only be necessary when we add MPI support
# export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
# export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"
# export MPICH_GPU_SUPPORT_ENABLED=1
# export MPICH_SMP_SINGLE_COPY_MODE=CMA
#export HIPFLAGS="-I$MPICH_DIR/include"
#export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64"
