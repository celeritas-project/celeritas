#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run Preshower benchmark with or without Celeritas offload
#
# Usage:
#   ./run-preshower.sh [celeritas|geant4] [additional ddsim options]
#
# Examples:
#   ./run-preshower.sh celeritas --numberOfEvents 1000
#   ./run-preshower.sh geant4 --random.seed=42
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

# Parse mode argument
MODE=$1
if [ "$MODE" != "celeritas" ] && [ "$MODE" != "geant4" ]; then
  log "Usage: $0 [celeritas|geant4] [additional ddsim options]"
  log ""
  log "  celeritas - Run with Celeritas offload (default)"
  log "  geant4    - Run with Geant4 only"
  exit 1
fi
shift  # Remove first argument

# Disable Celeritas if running in geant4 mode
if [ "$MODE" = "geant4" ]; then
  export CELER_DISABLE=1
fi

EXAMPLE_DIR=$(cd "$(dirname $0)" && pwd)

if [ -z "${Celeritas_ROOT}" ]; then
  Celeritas_ROOT=$(cd "$EXAMPLE_DIR"/../.. && pwd)/install
  log "warning: Celeritas_ROOT is undefined: using ${Celeritas_ROOT}"
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

CELER_LIB_DIR=$(ls -1 -d "$Celeritas_ROOT"/lib 2>/dev/null | head -1)
if [ -z "$CELER_LIB_DIR" ]; then
  log "error: celeritas installation not found inside $Celeritas_ROOT"
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

# Set output file name based on run mode
log "Running with ${MODE} physics"
output_file="preshower.root"
log "info: Output will be written to ${output_file}"

# Create mode-specific subdirectory and change to it
mkdir -p "${EXAMPLE_DIR}/output/${MODE}"
cd "${EXAMPLE_DIR}/output/${MODE}"

set -x
exec "$PYTHON" "$DDSIM" \
  --compactFile="${EXAMPLE_DIR}/input/Preshower.xml" \
  --steering="${EXAMPLE_DIR}/input/steeringFile.py" \
  --outputFile="${output_file}" \
  "$@"
