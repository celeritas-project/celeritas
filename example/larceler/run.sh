#!/bin/sh -ex
#-------------------------------- -*- sh -*- ---------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# This script runs the same steps as described in example/larceler/README.rst.
#-----------------------------------------------------------------------------#

# Download and patch the geometry file for Celeritas execution
curl -O https://raw.githubusercontent.com/DUNE/dunecore/b8fb0006da5ca878732740c2030cdac66699c3fc/dunecore/Geometry/gdml/dune10kt_v6_refactored_1x2x6.gdml
patch -p0 -o dune10kt_v6_refactored_1x2x6_zwires.gdml < dune10kt_v6_refactored_1x2x6_zwires.patch

# Run a single GENIE event
lar -c prodgenie_nu_dune10kt_1x2x6.fcl -n 1 -o genie-output.root

# Run LArG4 + IonAndScint with original dune10kt_v6_refactored_1x2x6.gdml
lar -c larg4_dune10kt_1x2x6.fcl -s genie-output.root -o larg4-output.root

# Run FastSim and Celeritas CPU optical simulations with patched geometry
lar -c opticalsim_dune10kt_1x2x6.fcl -s larg4-output.root -o fastsim-output.root
lar -c opticalsim_celeritas_dune10kt_1x2x6.fcl -s larg4-output.root -o celeritas-output.root

# Generate analysis files
lar -c pdsimana_job.fcl -s $PWD/fastsim-output.root
lar -c pdsimana_job.fcl -s $PWD/celeritas-output.root
