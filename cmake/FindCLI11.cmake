#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindCLI11
--------

Find the CLI11 Geant4/VecGeom adapter library.

#]=======================================================================]

find_package(CLI11 QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLI11 CONFIG_MODE)

#-----------------------------------------------------------------------------#
