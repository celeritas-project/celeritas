#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findcetmodules
--------------

Find CMake packages for LArSoft environment.

#]=======================================================================]

find_package(${CMAKE_FIND_PACKAGE_NAME} QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME} CONFIG_MODE)

#-----------------------------------------------------------------------------#
