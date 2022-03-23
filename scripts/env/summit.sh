#!/bin/sh -e

_celer_base=$PROJWORK/csc404/celeritas

module load DefApps/default cuda/11.0 gcc/9.1
module unload python
export PATH=${_celer_base}/spack/view/bin:$PATH
export CMAKE_PREFIX_PATH=${_celer_base}/spack/view:${CMAKE_PREFIX_PATH}
export MODULEPATH=/ccs/proj/csc404/spack/share/spack/lmod/linux-rhel8-ppc64le/gcc/9.1.0:$MODULEPATH
module load geant4-data
