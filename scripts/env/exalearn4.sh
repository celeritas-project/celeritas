#!/bin/sh -e

function _fail {
    echo "$1" >&2
    exit $2
}

celeritas_spack_env_name=celeritas

if ! declare -fF spack > /dev/null; then
    _fail "Expects spack shell support" 1
elif [[ ! -d "${SPACK_ROOT}/var/spack/environments/${celeritas_spack_env_name}" ]]; then
    _fail "Expects a spack environment named ${celeritas_spack_env_name}" 2
fi

unset LD_LIBRARY_PATH
module load gcc/12.1.0 ninja-build/1.10.1 git/2.36.1
spack env activate ${celeritas_spack_env_name}
