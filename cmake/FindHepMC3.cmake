#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC and other Celeritas developers.
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

# HepMC3's cmake config file does not currently provide an imported target.
if(NOT TARGET HepMC3::HepMC3)
  add_library(HepMC3::HepMC3 UNKNOWN IMPORTED)
  set_target_properties(HepMC3::HepMC3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HEPMC3_INCLUDE_DIR}"
    IMPORTED_LOCATION "${HEPMC3_LIB}"
  )
endif()

#-----------------------------------------------------------------------------#
