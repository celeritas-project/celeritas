#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

log() {
  printf "%s\n" "$1" >&2
}

# Resolve symlink chain to actual file/executable
resolve_symlinks() {
  _path="$1"
  while [ -L "$_path" ]; do
    printf "Following symlink: %s -> " "$_path" >&2
    _path=$(readlink -f "$_path")
    log "$_path"
  done
  printf "%s\n" "$_path"
}

EXAMPLE_DIR=$(cd "$(dirname $0)" && pwd)

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$EXAMPLE_DIR"/../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
  log "warning: CELER_INSTALL_DIR is undefined: using ${CELER_INSTALL_DIR}"
fi

# Resolve ddsim
DDSIM=$(command -v "ddsim" 2>/dev/null || printf "")
if [ -z "$DDSIM" ]; then
  log "error: ddsim: command not found"
  exit 1
fi
DDSIM=$(resolve_symlinks "$DDSIM" true)

if [ -z "$DD4hepINSTALL" ]; then
  log "error: DD4hepINSTALL environment variable is not set"
  log "  You must load DD4HEP's environment (including its PYTHONPATH and ROOT's)"
  exit 1
fi

CELER_LIB_DIR=$(ls -1 -d "$CELER_INSTALL_DIR"/lib | head -1)
if [ -z "$CELER_LIB_DIR" ]; then
  log "error: celeritas installation not found inside $CELER_INSTALL_DIR"
  exit 1
fi

# Plugin must be available in the runtime library path for DD4hep to find it
if [ "$(uname -s)" = "Darwin" ]; then
  _ld_prefix=DY
  export DYLD_LIBRARY_PATH=${CELER_LIB_DIR}:${DYLD_LIBRARY_PATH}
  if [ -z "$DD4HEP_LIBRARY_PATH" ]; then
    # Needed by dd4hep load on macos
    log "error: DD4HEP_LIBRARY_PATH environment variable is not set"
    exit 1
  fi
else
  _ld_prefix=
  export LD_LIBRARY_PATH=${CELER_LIB_DIR}:${LD_LIBRARY_PATH}
fi
log "info: Prepended ${CELER_LIB_DIR} to ${_ld_prefix}LD_LIBRARY_PATH"

# Find python interpreter (prefer python3)
PYTHON=$(command -v "python3" 2>/dev/null || command -v "python" 2>/dev/null || printf "")
if [ -z "$PYTHON" ]; then
  log "error: failed to find python3 or python"
  exit 1
fi
PYTHON=$(resolve_symlinks "$PYTHON" true)

set -x
exec "$PYTHON" "$DDSIM" \
  --compactFile=${EXAMPLE_DIR}/SiD_ConstantField.xml \
  --steering=${EXAMPLE_DIR}/steeringFile.py \
  $@
