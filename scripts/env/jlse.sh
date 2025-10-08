#!/bin/sh -e

module use /soft/modulefiles
export PROJ=/vast/projects/celeritas
export SPACKROOT=$PROJ/spack
. $SPACKROOT/share/spack/setup-env.sh

qsub-gpu() {
    # Usage: qsub-gpu [mi300x|h100] <qsub args>
    [[ $(hostname) =~ ^jlselogin[0-9]+ ]] || {
        echo "Error: must be on a jlselogin* host to submit jobs."
        return 1
    }

    case "$1" in
        mi300x) queue="gpu_amd_mi300x" ;;
        h100)   queue="gpu_h100" ;;
        *)      echo "Error: unknown GPU type '$1'"; return 1 ;;
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
