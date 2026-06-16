#!/bin/sh

set -ex

# Download and patch
curl -o dune10kt_v6_refactored_1x2x6_optical.gdml https://raw.githubusercontent.com/DUNE/dunecore/b8fb0006da5ca878732740c2030cdac66699c3fc/dunecore/Geometry/gdml/dune10kt_v6_refactored_1x2x6.gdml
patch -p1 < dune10kt-optical.patch
patch -p1 < dune10kt-deletezwires.patch
patch -p1 < dune10kt-addopdet.patch
