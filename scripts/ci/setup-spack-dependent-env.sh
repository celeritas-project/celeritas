#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Usage: sh setup-spack-dependent-env.sh CORE ENV VIEW SPEC [SPEC ...]
# Copy core configuration and include its concrete specs without modifying it.
# The dependent environment is always solved against the current core lockfile.
#-----------------------------------------------------------------------------#

set -eu

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 CORE ENV VIEW SPEC [SPEC ...]" >&2
  exit 1
fi

core_env=$(cd "$1" && pwd)
dependent_env=$2
dependent_view=$3
shift 3

spack env create -d "$dependent_env" "$core_env/spack.yaml" \
  --with-view "$dependent_view" --include-concrete "$core_env"
# The included roots already carry the exact core specs. Only the new packages
# belong in this environment's abstract spec list.
spack -e "$dependent_env" remove --all
spack -e "$dependent_env" config add "concretizer:unify:true"
spack -e "$dependent_env" config add "config:install_tree:padded_length:False"
spack -e "$dependent_env" add "$@"
