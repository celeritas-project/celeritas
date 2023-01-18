#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2023 UT-Battelle, LLC and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindGeant4
----------

Find the Geant4 HEP library.

#]=======================================================================]

# Save and restore global settings changed by Geant's find script
get_directory_property(_include_dirs INCLUDE_DIRECTORIES)

find_package(Geant4 QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Geant4 CONFIG_MODE)

if(Geant4_FOUND)
  # Geant4 calls `include_directories` for CLHEP :( which is not what we want!
  # Save and restore include directories around the call -- even though as a
  # standalone project Celeritas will never have directory-level includes
  set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${_include_dirs}")
endif()
unset(_include_dirs)

if(Geant4_FOUND AND Geant4_VERSION VERSION_GREATER_EQUAL 11 AND CELERITAS_USE_CUDA)
  target_compile_features(Geant4::G4global INTERFACE cuda_std_17)
endif()

#-----------------------------------------------------------------------------#
