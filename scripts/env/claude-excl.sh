#!/bin/sh -ex
# Local, untracked: secretariat settings using the dunesw-cuda spack view.

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

CELER_SPACK_VIEW=/auto/projects/celeritas/view/dunesw-cuda
if ! [ -d "${CELER_SPACK_VIEW}" ]; then
  echo "error: missing view ${CELER_SPACK_VIEW}" >&2
  return 1
fi
export CELER_SPACK_VIEW
export PATH=${CELER_SPACK_VIEW}/bin:${PATH}
export CMAKE_PREFIX_PATH=${CELER_SPACK_VIEW}:${CMAKE_PREFIX_PATH}
export CUDACXX=${CELER_SPACK_VIEW}/bin/nvcc
export PKG_CONFIG_PATH=/auto/projects/celeritas/view/dunesw-cuda/lib/pkgconfig:/auto/projects/celeritas/view/dunesw-cuda/lib64/pkgconfig:/auto/projects/celeritas/view/dunesw-cuda/share/pkgconfig:${PKG_CONFIG_PATH}
