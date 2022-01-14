#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindGeant4
----------

Find the Geant4 HEP library.

#]=======================================================================]

find_package(Geant4 QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Geant4 CONFIG_MODE)

#-----------------------------------------------------------------------------#
