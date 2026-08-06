export SPACK_ROOT=/projects/spack
CELER_SPACK_VIEW=${SPACK_ROOT}/var/spack/environments/celeritas/.spack-env/view
export MODULEPATH=${SPACK_ROOT}/share/spack/lmod/linux-rhel8-x86_64/Core
export PATH=${CELER_SPACK_VIEW}/bin:/usr/local/cuda-12.9/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELER_SPACK_VIEW}:${CMAKE_PREFIX_PATH}
