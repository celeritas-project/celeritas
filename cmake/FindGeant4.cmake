#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC and other Celeritas developers.
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

if(Geant4_FOUND AND Geant4_VERSION VERSION_GREATER_EQUAL 11 AND CELERITAS_USE_CUDA)
  foreach(_tgt Geant4::G4global Geant4::G4global-static)
    if(TARGET ${_tgt})
      target_compile_features(${_tgt} INTERFACE cuda_std_17)
    endif()
  endforeach()
endif()

if(Geant4_VERSION VERSION_LESS 10.6)
  # Version 10.5 and older have some problems.

  # Move definitions from CXX flags to geant4 definitions
  string(REGEX MATCHALL "-D[a-zA-Z0-9_]+" _defs "${Geant4_CXX_FLAGS}")
  list(APPEND Geant4_DEFINITIONS ${_defs})
  unset(defs)

  # Make a fake target with includes and definitions
  set(_tgt Geant4_headers)
  if(NOT TARGET "${_tgt}")
    add_library(${_tgt} INTERFACE)
    add_library(celeritas::${_tgt} ALIAS ${_tgt})
    target_include_directories(${_tgt} INTERFACE ${Geant4_INCLUDE_DIRS})
    target_compile_definitions(${_tgt} INTERFACE ${Geant4_DEFINITIONS})
    install(TARGETS ${_tgt} EXPORT celeritas-targets)
  endif()
  # Add the fake target to the list of geant4 libraries
  list(APPEND Geant4_LIBRARIES ${_tgt})
  unset(_tgt)
endif()
if(Geant4_FOUND)
  # Geant4 calls `include_directories` for CLHEP :( which is not what we want!
  # Save and restore include directories around the call -- even though as a
  # standalone project Celeritas will never have directory-level includes
  set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${_include_dirs}")
endif()
unset(_include_dirs)

#-----------------------------------------------------------------------------#