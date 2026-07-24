#!/bin/sh -e
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# This script runs the same steps as described in example/larceler/README.rst.
#-----------------------------------------------------------------------------#
EXAMPLE_DIR="$(cd "$(dirname $0)" && pwd)"

# Check that Celeritas
if [ -z "${CELERITAS_DIR}" ]; then
  echo "error: Celeritas has not been activated: run 'eval \$(\$CELERITAS_DIR/bin/larceler-env)'"
  exit 1
fi

set -x

# Set up environment:
# - for spack-based execution, the working directory is *not* included
#   in the FHICL lookup path
export FHICL_FILE_PATH=$FHICL_FILE_PATH:$EXAMPLE_DIR
# - the modified GDML lookup also requires the current directory (perhaps only for
#   newer LArSoft versions?)
export FW_SEARCH_PATH=.:$FW_SEARCH_PATH

# Download and patch the geometry file for Celeritas execution into the current directory
if ! [ -f dune10kt_v6_refactored_1x2x6_optical.gdml ]; then
  sh "${EXAMPLE_DIR}/setup-dune-gdml.sh"
fi

# Run GENIE (number of events can be passed via the first argument to this script)
NUM_EVENTS=${1:-1}
lar -c prodgenie_nu_dune10kt_1x2x6.fcl -n $NUM_EVENTS -o genie-output-${NUM_EVENTS}.root

# Run LArG4 + IonAndScint with original dune10kt_v6_refactored_1x2x6.gdml
lar -c "${EXAMPLE_DIR}/larg4_dune10kt_1x2x6.fcl" -s genie-output-${NUM_EVENTS}.root -o larg4-output.root

# Run FastSim and Celeritas CPU optical simulations with patched geometry
lar -c "${EXAMPLE_DIR}/opticalsim_dune10kt_1x2x6.fcl" -s larg4-output.root -o fastsim-output.root
lar -c "${EXAMPLE_DIR}/opticalsim_celeritas_dune10kt_1x2x6.fcl" -s larg4-output.root -o celeritas-output.root

# Generate analysis files
lar -c pdsimana_job.fcl -s $PWD/fastsim-output.root
lar -c pdsimana_job.fcl -s $PWD/celeritas-output.root
