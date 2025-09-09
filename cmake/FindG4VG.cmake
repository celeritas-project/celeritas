#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindG4VG
--------

Find the G4VG Geant4/VecGeom adapter library.

#]=======================================================================]

find_package(G4VG QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(G4VG CONFIG_MODE)

#-----------------------------------------------------------------------------#
