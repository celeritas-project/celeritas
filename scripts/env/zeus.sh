#!/bin/sh -e

function _fail {
    echo "$1" >&2
    exit $2
}

CELER_SPACK_ENV=celeritas

if ! declare -fF spack > /dev/null; then
    _fail "Expects spack shell support" 1
elif [[ ! -d "${SPACK_ROOT}/var/spack/environments/${CELER_SPACK_ENV}" ]]; then
    _fail "Expects a spack environment named ${CELER_SPACK_ENV}" 2
fi

module unload gcc
source /cvmfs/sft.cern.ch/lcg/contrib/gcc/11.3.0/x86_64-centos7-gcc11-opt/setup.sh
module load cuda/11.8.0
spack env activate ${CELER_SPACK_ENV}
