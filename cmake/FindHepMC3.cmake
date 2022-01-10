#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindHepMC3
----------

Find the HepMC3 HEP I/O library.

#]=======================================================================]

find_package(HepMC3 QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HepMC3 CONFIG_MODE)

#-----------------------------------------------------------------------------#
