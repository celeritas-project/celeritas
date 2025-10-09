#!/bin/sh -e

module use /soft/modulefiles
export PROJ=/vast/projects/celeritas
export SPACKROOT=$PROJ/spack
. $SPACKROOT/share/spack/setup-env.sh

qsub_gpu() {
    # Usage: qsub_gpu [mi300x|h100] <qsub args>

    # Verify we are on a login node
    h=$(uname -n 2>/dev/null || hostname 2>/dev/null || printf '')
    if ! expr "$h" : 'jlselogin[0-9][0-9]*$' >/dev/null 2>&1; then
        printf '%s\n' 'Error: must be on a login node to submit jobs.' >&2
        return 1
    fi

    case ${1-} in
        mi300x) queue=gpu_amd_mi300x ;;
        h100)   queue=gpu_h100 ;;
        *)      printf "Error: unknown GPU type '%s'\n" "${1-}" >&2; return 1 ;;
    esac

    shift
    exec qsub -q "$queue" "$@"
}

# Load Nvidia/AMD environment if on a compute node
if [[ "$(hostname -s)" == hopper* ]]; then
  echo "Loading environment for H100"
  module purge
  module load cmake/3.28.3
  module load cuda/12.9.1
  spacktivate h100
  export ENVFILE="$SPACKROOT/var/spack/environments/h100/spack.yaml"
elif [[ "$(hostname -s)" == amdgpu* ]]; then
  echo "Loading environment for MI300X"
  module purge
  module load cmake/3.28.3
  module load aomp/rocm-6.4.1
  spacktivate mi300x
  export ENVFILE="$SPACKROOT/var/spack/environments/mi300x/spack.yaml"
fi
