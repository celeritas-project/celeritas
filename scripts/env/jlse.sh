#!/bin/sh -e

module use /soft/modulefiles
export PROJ=/vast/projects/celeritas
export SPACKROOT=/vast/projects/celeritas/spack
. $SPACKROOT/share/spack/setup-env.sh

# Interactive job on an MI30
alias mi300x="qsub -I -t 60 -n 1 -q gpu_amd_mi300x"

# Interactive job on an H100
alias h100="qsub -I -t 60 -n 1 -q gpu_h100"

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
