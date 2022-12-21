#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

include(CeleritasGen)

#----------------------------------------------------------------------------#
# CONFIGURE FILES
#----------------------------------------------------------------------------#
# Celeritas version

celeritas_configure_file("celeritas_version.h.in" "celeritas_version.h" @ONLY)

#----------------------------------------------------------------------------#
# Build configuration

set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

# Define a numeric table of options for the default runtime RNG.
# Start counter from 1 because undefined macros have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
set(CELERITAS_RNG_MACROS)
set(_rng_counter 1)
foreach(_rng IN LISTS CELERITAS_RNG_OPTIONS)
  list(APPEND CELERITAS_RNG_MACROS
    "#define CELERITAS_RNG_${_rng} ${_rng_counter}"
  )
  math(EXPR _rng_counter "${_rng_counter} + 1")
endforeach()

string(JOIN "\n" CELERITAS_RNG_MACROS
  ${CELERITAS_RNG_MACROS}
  "#define CELERITAS_RNG CELERITAS_RNG_${CELERITAS_RNG}"
)
celeritas_configure_file("celeritas_config.h.in" "celeritas_config.h" @ONLY)

#----------------------------------------------------------------------------#
# Detailed build configuration for reproducibility/provenance

# Get host system information
if(NOT DEFINED CELERITAS_HOSTNAME)
  set(_hostname "$ENV{LMOD_SYSTEM_NAME}")
  if(NOT _hostname)
    cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
    string(REGEX REPLACE "\\..*" "" _hostname "${_hostname}")
  endif()
  set(CELERITAS_HOSTNAME "${_hostname}" CACHE INTERNAL
    "Build/deploy system name")
endif()

set(CELERITAS_CMAKE_STRINGS)
set(CELERITAS_BUILD_TYPE ${CMAKE_BUILD_TYPE})
foreach(_var CELERITAS_BUILD_TYPE CELERITAS_HOSTNAME CELERITAS_RNG)
  string(TOLOWER "${_var}" _lower)
  string(APPEND CELERITAS_CMAKE_STRINGS
    "static const char ${_lower}[] = \"${${_var}}\";\n"
  )
endforeach()

foreach(_var CLHEP_VERSION Geant4_VERSION VecGeom_VERSION)
  string(TOLOWER "${_var}" _lower)
  string(APPEND CELERITAS_CMAKE_STRINGS
    "static const char celeritas_${_lower}[] = \"${${_var}}\";\n"
  )
endforeach()
celeritas_configure_file("celeritas_cmake_strings.h.in" "celeritas_cmake_strings.h" @ONLY)

#----------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------#

if(BUILD_SHARED_LIBS OR CELERITAS_USE_ROOT)
  # Make sure modules, sub-libraries, etc. are relocatable
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

add_subdirectory(corecel)
add_subdirectory(orange)
add_subdirectory(celeritas)
if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
endif()

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_Geant4)
  celeritas_target_include_directories(celeritas
    PUBLIC
    $<BUILD_INTERFACE:${Geant4_INCLUDE_DIRS}>
  )
endif()

#----------------------------------------------------------------------------#
# HEADERS
#----------------------------------------------------------------------------#

# C++ source headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
  FILES_MATCHING REGEX ".*\\.hh?$"
)

#---------------------------------------------------------------------------##
