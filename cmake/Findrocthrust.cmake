#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findrocthrust
-------------

Find the ROCm version of the thrust library.

#]=======================================================================]

find_package(rocthrust QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocthrust CONFIG_MODE)

#-----------------------------------------------------------------------------#
