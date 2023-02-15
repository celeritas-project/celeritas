#!/bin/sh -e

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
# we need to clear exit on unchecked error for atlasLocalSetup...
set +e
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup none,gcc11,cmakesetup,siteroot=cvmfs
lsetup "views LCG_102b_ATLAS_11 x86_64-centos7-gcc11-opt"
set -e
module load cuda/11.8.0
source /bld3/build/celeritas/geant4/geant4-v11.1.0-install/bin/geant4.sh