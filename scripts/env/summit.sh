#!/bin/sh -e

_celer_base=$PROJWORK/csc404/celeritas

module load DefApps/default cuda/11.0 gcc/9.1
module unload python
export PATH=${_celer_base}/spack/view/bin:$PATH
export CMAKE_PREFIX_PATH=${_celer_base}/spack/view:${CMAKE_PREFIX_PATH}
export MODULEPATH=/ccs/proj/csc404/spack/share/spack/lmod/linux-rhel8-ppc64le/gcc/9.1.0:$MODULEPATH
module load geant4-data


# NOTE: these might only be necessary when we add MPI support
# export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
# export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"
# export MPICH_GPU_SUPPORT_ENABLED=1
# export MPICH_SMP_SINGLE_COPY_MODE=CMA
#export HIPFLAGS="-I$MPICH_DIR/include"
#export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64"
