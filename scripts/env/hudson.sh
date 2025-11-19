#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

export SPACK_ROOT=/auto/projects/celeritas/spack
export CXX=/usr/bin/c++

# Reduce I/O metadata overhead by avoiding language translation lookups
export LC_ALL=C

if ! test -r $SPACK_ROOT; then
  printf "Error: spack directory %s is not readable:
       contact excl-help@ornl.gov for access\n" "${SPACK_ROOT}" >&2
  exit 1
fi

if ! command -v spack >/dev/null 2>&1; then
  _spack_setup="$SPACK_ROOT/share/spack/setup-env.sh"
  case "$0" in
    /bin/*)
      # Presumably sourcing from user shell to set up environment: load spack
      printf "Loading spack setup from %s\n" "${_spack_setup}" >&2
      . "${_spack_setup}"
      ;;
    *)
      # Spack isn't available but we're loading from the build script so we don't really need it
      printf "Warning: spack is not available; source %s if desired\n" "${_spack_setup}" >&2
      ;;
  esac
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
  printf "Creating clangd config: %s\n" "${_clangd}" >&2
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
