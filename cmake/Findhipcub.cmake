#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findhipcub
--------

Find the hipcub library.

#]=======================================================================]

find_package(hipcub QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hipcub CONFIG_MODE)

#-----------------------------------------------------------------------------#
