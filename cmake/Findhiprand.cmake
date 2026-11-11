#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findhiprand
--------

Find the hiprand library.

#]=======================================================================]

find_package(hiprand QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hiprand CONFIG_MODE)

#-----------------------------------------------------------------------------#
