#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindThrust
----------

Find the Thrust algorithm library for CUDA. Note that HIP's installation may be
available under the name "rocthrust" but we can't handle that.

#]=======================================================================]

find_package(Thrust QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust CONFIG_MODE)

#-----------------------------------------------------------------------------#
