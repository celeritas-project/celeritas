#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findcovfie
---------

Find the Covfie magnetic field interpolation library.

#]=======================================================================]

find_package(covfie QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(covfie CONFIG_MODE)

#-----------------------------------------------------------------------------#