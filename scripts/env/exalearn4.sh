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

unset LD_LIBRARY_PATH
module load gcc/12.1.0 ninja-build/1.10.1 git/2.36.1
spack env activate ${CELER_SPACK_ENV}
