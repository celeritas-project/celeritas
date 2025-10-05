export SPACK_ROOT=/projects/spack
CELERITAS_ENV=${SPACK_ROOT}/var/spack/environments/celeritas/.spack-env/view
export MODULEPATH=${SPACK_ROOT}/share/spack/lmod/linux-rhel8-x86_64/Core
export PATH=${CELERITAS_ENV}/bin:/usr/local/cuda-12.9/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELERITAS_ENV}:${CMAKE_PREFIX_PATH}
