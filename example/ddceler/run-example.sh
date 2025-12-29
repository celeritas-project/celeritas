#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

EXAMPLE_DIR=$(cd "$(dirname $0)" && pwd)

if [ -z "${CELER_SOURCE_DIR}" ]; then
  CELER_SOURCE_DIR=$(cd "$EXAMPLE_DIR"/../.. && pwd)
fi
if [ -z "${CELER_INSTALL_DIR}" ]; then
  CELER_INSTALL_DIR="${CELER_SOURCE_DIR}/install"
  echo "CELER_INSTALL_DIR is undefined: using ${CELER_INSTALL_DIR}"
fi

DDSIM=$(command -v "ddsim" 2>/dev/null || printf "")
if [ -z "$DDSIM" ]; then
  echo "ddsim: command not found"
  exit 1
fi

CELER_LIB_DIR=$(ls -1 -d "$CELER_INSTALL_DIR"/lib | head -1)
if [ -z "$CELER_LIB_DIR" ]; then
  echo "celeritas installation not found inside $CELER_INSTALL_DIR"
  exit 1
fi

# Plugin must be available in the runtime library path for DD4hep to find it
if [ "$(uname -s)" = "Darwin" ]; then
  _ld_var=DYLD_LIBRARY_PATH
else
  _ld_var=LD_LIBRARY_PATH
fi
echo "Prepending ${CELER_LIB_DIR} to ${_ld_var}"
export ${_ld_var}=${CELER_LIB_DIR}:${_ld_var}

exec ddsim \
  --compactFile=${EXAMPLE_DIR}/SiD_ConstantField.xml \
  --steering=${EXAMPLE_DIR}/steeringFile.py \
  $@
