#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

export SPACK_ROOT=/auto/projects/celeritas/spack
export CXX=/usr/bin/c++

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

if ! command -v spack >/dev/null 2>&1; then
  . $SPACK_ROOT/share/spack/setup-env.sh
fi

for _d in build install ccache; do
  # Create build/install in higher-performance local-but-persistent dir
  _scratch="/scratch/$USER/$_d"
  if ! test -d "${_scratch}"; then
    printf "Creating scratch directory at %s\n" "${_scratch}" >&2
    mkdir -p "${_scratch}"
    chmod 700 "${_scratch}"
  fi
done

_clangd="$GIT_WORK_TREE/.clangd"
if [ -n "$GIT_WORK_TREE" ] && [ ! -e "${_clangd}" ]; then
  # Create clangd compatible with the system and build config
  printf "Creating clangd config: %s\n" "${_scratch}" >&2
  cat > "${_clangd}" << EOF
CompileFlags:
  CompilationDatabase: /scratch/s3j/build/celeritas-reldeb
  Add:
    [
      -isystem,
      /usr/include/c++/13,
      -isystem,
      /usr/local/include,
      -isystem,
      /usr/include,
      -isystem,
      /usr/include/x86_64-linux-gnu/c++/13,
    ]
EOF
fi

CELERITAS_ENV=${SPACK_ROOT}/var/spack/environments/celeritas/.spack-env/view
export PATH=${CELERITAS_ENV}/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELERITAS_ENV}:${CMAKE_PREFIX_PATH}

export CCACHE_DIR=/scratch/$USER/ccache
