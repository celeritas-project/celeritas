#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# This script runs the same steps as described in example/larceler/README.rst.
#
# The first argument to the script is the number of events to send to GENIE.
#-----------------------------------------------------------------------------#
EXAMPLE_DIR="$(cd "$(dirname $0)" && pwd)"

# Check that Celeritas library/FHiCL paths are loaded
if [ -z "${CELERITAS_DIR}" ]; then
  echo "error: Celeritas has not been activated: run 'eval \$(\$CELERITAS_DIR/bin/larceler-env)'"
  exit 1
fi

set -x

# Set up environment:
# - Unlike with UPS, Spack-based LArSoft/DUNESW environment
#   does *not* automatically include the working directory
#   in the FHICL lookup path
# - the modified GDML lookup also requires the example directory
export FHICL_FILE_PATH=$EXAMPLE_DIR:$FHICL_FILE_PATH
# - the patched GDML is downloaded into the current directory, so that must
#   be searched too
export FW_SEARCH_PATH=$PWD:$EXAMPLE_DIR:$FW_SEARCH_PATH

# Download and patch the geometry file for Celeritas execution into the current directory
if ! [ -f dune10kt_v6_refactored_1x2x6_optical.gdml ]; then
  sh "${EXAMPLE_DIR}/setup-dune-gdml.sh"
fi

# Get event count from first argument (default 1)
NUM_EVENTS=${1:-1}
# Note: this fcl file is part of the `dunesw` spack package
# Some GENIE libraries use libGFwReg symbols without linking against it, which
# fails with an undefined symbol unless it is loaded first:
#   https://github.com/GENIE-MC/Generator/issues/570
# Preload it for the GENIE run only, if the Spack view provides it.
genie_ld_preload=${LD_PRELOAD}
if [ -n "${SPACK_ENV_VIEW}" ] \
   && genie_reg=$(realpath "${SPACK_ENV_VIEW}/lib/libGFwReg.so" 2>/dev/null) \
   && [ -x "${genie_reg}" ]; then
  genie_ld_preload="${genie_reg}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
LD_PRELOAD="${genie_ld_preload}" \
  lar -c prodgenie_nu_dune10kt_1x2x6.fcl -n $NUM_EVENTS -o genie-output-${NUM_EVENTS}.root

# Run LArG4 + IonAndScint with original dune10kt_v6_refactored_1x2x6.gdml
lar -c "${EXAMPLE_DIR}/larg4_dune10kt_1x2x6.fcl" -s genie-output-${NUM_EVENTS}.root -o larg4-output.root

# Run FastSim and Celeritas CPU optical simulations with patched geometry
lar -c "${EXAMPLE_DIR}/opticalsim_dune10kt_1x2x6.fcl" -s larg4-output.root -o fastsim-output.root
lar -c "${EXAMPLE_DIR}/opticalsim_celeritas_dune10kt_1x2x6.fcl" -s larg4-output.root -o celeritas-output.root

# Generate analysis files
lar -c pdsimana_job.fcl -s $PWD/fastsim-output.root
lar -c pdsimana_job.fcl -s $PWD/celeritas-output.root
