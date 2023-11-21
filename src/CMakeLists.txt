#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# CONFIGURE FILES
#----------------------------------------------------------------------------#
# Celeritas version

# Add an extra 1 up front and strip it to zero-pad
math(EXPR _temp_version
  "((256 + ${PROJECT_VERSION_MAJOR}) * 256 + ${PROJECT_VERSION_MINOR}) * 256 + ${PROJECT_VERSION_PATCH}"
  OUTPUT_FORMAT HEXADECIMAL
)
string(SUBSTRING "${_temp_version}" 3 -1 _temp_version)
set(CELERITAS_VERSION "0x${_temp_version}")
celeritas_configure_file("celeritas_version.h.in" "celeritas_version.h" @ONLY)

#----------------------------------------------------------------------------#
# Build configuration

set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

# Set CELERITAS_CORE_RNG_CONFIG to a numeric table of RNG options and the
# selection.
# Start counter from 1 because undefined config have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
celeritas_generate_option_config(CELERITAS_CORE_RNG)
# Same, but with CELERITAS_CORE_GEO_CONFIG
celeritas_generate_option_config(CELERITAS_CORE_GEO)
# Same, but with CELERITAS_REAL_TYPE
celeritas_generate_option_config(CELERITAS_REAL_TYPE)
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
foreach(_var BUILD_TYPE HOSTNAME REAL_TYPE CORE_GEO CORE_RNG)
  set(_var "CELERITAS_${_var}")
  string(TOLOWER "${_var}" _lower)
  string(APPEND CELERITAS_CMAKE_STRINGS
    "static const char ${_lower}[] = \"${${_var}}\";\n"
  )
endforeach()

foreach(_var CLHEP_VERSION Geant4_VERSION VecGeom_VERSION)
  string(TOLOWER "${_var}" _lower)
  if(DEFINED ${_var}_STRING)
    set(_var "${_var}_STRING")
  endif()
  set(_value "${${_var}}")
  string(APPEND CELERITAS_CMAKE_STRINGS
    "static const char celeritas_${_lower}[] = \"${_value}\";\n"
  )
endforeach()
celeritas_configure_file("celeritas_cmake_strings.h.in" "celeritas_cmake_strings.h" @ONLY)

celeritas_configure_file("celeritas_sys_config.h.in" "celeritas_sys_config.h" @ONLY)

#----------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------#

add_subdirectory(corecel)
add_subdirectory(orange)
add_subdirectory(celeritas)
if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
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
