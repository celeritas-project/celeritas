#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Run the optical photon demonstration with or without Celeritas offload.
#
# Usage:
#   ./run.sh [celeritas|geant4] [additional ddsim options]
#
# Examples:
#   ./run.sh celeritas
#   ./run.sh celeritas --numberOfEvents 10
#   ./run.sh geant4
#-----------------------------------------------------------------------------#

log() { printf "%s\n" "$1" >&2; }

resolve_symlinks() {
  _path="$1"
  while [ -L "$_path" ]; do
    _path=$(readlink -f "$_path")
  done
  printf "%s\n" "$_path"
}

MODE=$1
if [ "$MODE" != "celeritas" ] && [ "$MODE" != "geant4" ]; then
  log "Usage: $0 [celeritas|geant4] [additional ddsim options]"
  exit 1
fi
shift

[ "$MODE" = "geant4" ] && export CELER_DISABLE=1

EXAMPLE_DIR=$(cd "$(dirname $0)" && pwd)

if [ -z "${Celeritas_ROOT}" ]; then
  Celeritas_ROOT=$(cd "$EXAMPLE_DIR"/../../.. && pwd)/install
  log "warning: Celeritas_ROOT is undefined: using ${Celeritas_ROOT}"
fi

DDSIM=$(command -v ddsim 2>/dev/null || printf "")
[ -z "$DDSIM" ] && { log "error: ddsim not found"; exit 1; }
DDSIM=$(resolve_symlinks "$DDSIM")

[ -z "$DD4hepINSTALL" ] && { log "error: DD4hepINSTALL not set"; exit 1; }

CELER_LIB_DIR=$(ls -1 -d "$Celeritas_ROOT"/lib 2>/dev/null | head -1)
[ -z "$CELER_LIB_DIR" ] && { log "error: celeritas not found in $Celeritas_ROOT"; exit 1; }

if [ "$(uname -s)" = "Darwin" ]; then
  export DYLD_LIBRARY_PATH=${CELER_LIB_DIR}:${DYLD_LIBRARY_PATH}
else
  export LD_LIBRARY_PATH=${CELER_LIB_DIR}:${LD_LIBRARY_PATH}
fi

PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || printf "")
[ -z "$PYTHON" ] && { log "error: python not found"; exit 1; }
PYTHON=$(resolve_symlinks "$PYTHON")

log "Running optical demo with ${MODE} physics"
mkdir -p "${EXAMPLE_DIR}/output/${MODE}"
cd "${EXAMPLE_DIR}/output/${MODE}"

set -x
exec "$PYTHON" "$DDSIM" \
  --compactFile="${EXAMPLE_DIR}/input/ScintWater.xml" \
  --steering="${EXAMPLE_DIR}/input/steeringFile.py" \
  --outputFile="optical_demo.root" \
  "$@"
