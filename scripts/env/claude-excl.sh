#!/bin/sh -ex
# Settings for apptainer-based Claude on ExCL: the dunesw spack view (CUDA or
# HIP) is selected by the env script that the user sourced beforehand.

export CELER_DISABLE_DEVICE=1
export CXX=/usr/bin/g++-13
export CC=/usr/bin/gcc-13
export CUDAARCHS="70;80;90"

export USER="${USER:-$(id -un)}"
export LC_ALL=C
# Locations come from the apptainer launcher (claude-run): TMPDIR and
# XDG_CACHE_HOME are bound to local scratch storage, as is /work/build (see the
# presets: build in ../build, install in ../install relative to the source).
if [ -z "${XDG_CACHE_HOME}" ] || [ -z "${TMPDIR}" ]; then
  echo "error: XDG_CACHE_HOME and TMPDIR must be set (run via claude-run)" >&2
  return 1
fi
mkdir -p "${XDG_CACHE_HOME}/ccache" || return $?
export CCACHE_DIR="${XDG_CACHE_HOME}/ccache"
export CCACHE_TEMPDIR="${TMPDIR}/ccache"
mkdir -p "${CCACHE_TEMPDIR}"
# Share cache hits regardless of source/build location
export CCACHE_BASEDIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export CCACHE_NOHASHDIR=1

# Require that the user has already sourced one of the view environment
# scripts (/auto/projects/celeritas/environments/scripts/dunesw-{cuda,hip}-env.sh).
# They set SPACK_ENV_VIEW and put the view on PATH, CMAKE_PREFIX_PATH,
# PKG_CONFIG_PATH, etc., along with FHICL_FILE_PATH and FW_SEARCH_PATH.
if [ -z "${SPACK_ENV_VIEW}" ]; then
  echo "error: source /auto/projects/celeritas/environments/scripts/dunesw-{cuda,hip}-env.sh first" >&2
  return 1
fi
if [ -x "${SPACK_ENV_VIEW}/bin/nvcc" ]; then
  export CUDACXX=${SPACK_ENV_VIEW}/bin/nvcc
fi
