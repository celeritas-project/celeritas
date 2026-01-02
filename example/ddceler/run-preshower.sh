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

# Resolve symlink chain to actual file/executable
resolve_symlinks() {
  _path="$1"
  while [ -L "$_path" ]; do
    if [ "$_verbose" = "true" ]; then
      printf "Following symlink: %s -> " "$_path"
    fi
    _path=$(readlink -f "$_path")
    if [ "$_verbose" = "true" ]; then
      printf "%s\n" "$_path"
    fi
  done
  echo "$_path"
}

EXAMPLE_DIR=$(cd "$(dirname $0)" && pwd)

# Parse mode argument
MODE="${1:-celeritas}"
if [ "$MODE" != "celeritas" ] && [ "$MODE" != "geant4" ]; then
  echo "Usage: $0 [celeritas|geant4] [additional ddsim options]"
  echo ""
  echo "  celeritas - Run with Celeritas offload (default)"
  echo "  geant4    - Run with Geant4 only"
  exit 1
fi
shift  # Remove first argument

# Disable Celeritas if running in geant4 mode
if [ "$MODE" = "geant4" ]; then
  export CELER_DISABLE=1
fi

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$EXAMPLE_DIR"/../../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
  echo "warning: CELER_INSTALL_DIR is undefined: using ${CELER_INSTALL_DIR}"
fi

# Resolve ddsim
DDSIM=$(command -v "ddsim" 2>/dev/null || printf "")
if [ -z "$DDSIM" ]; then
  echo "error: ddsim: command not found"
  exit 1
fi
DDSIM=$(resolve_symlinks "$DDSIM" true)

if [ -z "$DD4hepINSTALL" ]; then
  echo "error: DD4hepINSTALL environment variable is not set"
  echo "  You must load DD4HEP's environment (including its PYTHONPATH and ROOT's)"
  exit 1
fi

CELER_LIB_DIR=$(ls -1 -d "$CELER_INSTALL_DIR"/lib 2>/dev/null | head -1)
if [ -z "$CELER_LIB_DIR" ]; then
  echo "error: celeritas installation not found inside $CELER_INSTALL_DIR"
  exit 1
fi

# Plugin must be available in the runtime library path for DD4hep to find it
if [ "$(uname -s)" = "Darwin" ]; then
  _ld_prefix=DY
  export DYLD_LIBRARY_PATH=${CELER_LIB_DIR}:${DYLD_LIBRARY_PATH}
else
  _ld_prefix=
  export LD_LIBRARY_PATH=${CELER_LIB_DIR}:${LD_LIBRARY_PATH}
fi
echo "info: Prepended ${CELER_LIB_DIR} to ${_ld_prefix}LD_LIBRARY_PATH"

# Find python interpreter (prefer python3)
PYTHON=$(command -v "python3" 2>/dev/null || command -v "python" 2>/dev/null || printf "")
if [ -z "$PYTHON" ]; then
  echo "error: failed to find python3 or python"
  exit 1
fi
PYTHON=$(resolve_symlinks "$PYTHON" true)

echo "info: Running in ${MODE} mode"
echo "info: Output will be written to results/${MODE}/preshower-${MODE}.root"

# Create mode-specific subdirectory and change to it
mkdir -p "${EXAMPLE_DIR}/results/${MODE}"
cd "${EXAMPLE_DIR}/results/${MODE}"

set -x
exec "$PYTHON" "$DDSIM" \
  --compactFile="${EXAMPLE_DIR}/input/Preshower.xml" \
  --steering="${EXAMPLE_DIR}/input/steeringFile.py" \
  --outputFile="preshower-${MODE}.root" \
  "$@"
