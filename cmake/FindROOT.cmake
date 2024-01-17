#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindROOT
--------

Find the ROOT library.

#]=======================================================================]

find_package(ROOT QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROOT CONFIG_MODE)

#-----------------------------------------------------------------------------#
