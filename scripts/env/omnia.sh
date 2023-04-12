#!/bin/sh -e

_celeritas_view=/home/users/s3j/spack/var/spack/environments/celeritas-rocm/.spack-env/view

module load cmake rocm/5.5.0
export CMAKE_PREFIX_PATH=${_celeritas_view}:${CMAKE_PREFIX_PATH}
export PATH=${_celeritas_view}/bin:${PATH}
