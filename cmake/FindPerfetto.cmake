#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindPerfetto
------------

Find the Perfetto performance analysis library.

#]=======================================================================]

find_package(Perfetto QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Perfetto CONFIG_MODE)

#-----------------------------------------------------------------------------#
