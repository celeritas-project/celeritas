#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# CONFIGURE FILES
#----------------------------------------------------------------------------#
# Celeritas version

celeritas_version_to_hex(CELERITAS_VERSION PROJECT_VERSION)
celeritas_configure_file("celeritas_version.h.in" "celeritas_version.h" @ONLY)
celeritas_configure_file("corecel/Version.hh.in" "corecel/Version.hh" @ONLY)

#----------------------------------------------------------------------------#
# Build configuration

set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})
set(CELERITAS_USE_PERFETTO ${CELERITAS_USE_Perfetto})

# Set CELERITAS_CORE_RNG_CONFIG to a numeric table of RNG options and the
# selection.
# Start counter from 1 because undefined config have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
celeritas_generate_option_config(CELERITAS_CORE_GEO)
celeritas_generate_option_config(CELERITAS_CORE_RNG)
celeritas_generate_option_config(CELERITAS_OPENMP)
celeritas_generate_option_config(CELERITAS_REAL_TYPE)
celeritas_generate_option_config(CELERITAS_UNITS)
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
foreach(_var BUILD_TYPE HOSTNAME REAL_TYPE UNITS OPENMP CORE_GEO CORE_RNG)
  set(_var "CELERITAS_${_var}")
  string(TOLOWER "${_var}" _lower)
  string(APPEND CELERITAS_CMAKE_STRINGS
    "inline constexpr char ${_lower}[] = \"${${_var}}\";\n"
  )
endforeach()

foreach(_var CLHEP_VERSION Geant4_VERSION VecGeom_VERSION)
  string(TOLOWER "${_var}" _lower)
  if(DEFINED ${_var}_STRING)
    set(_var "${_var}_STRING")
  endif()
  set(_value "${${_var}}")
  string(APPEND CELERITAS_CMAKE_STRINGS
    "inline constexpr char celeritas_${_lower}[] = \"${_value}\";\n"
  )
endforeach()
celeritas_configure_file("celeritas_cmake_strings.hh.in" "celeritas_cmake_strings.hh" @ONLY)

# Save dependency versions to sys_config
set(CELERITAS_DEPENDENCY_VERSIONS)
foreach(_dep Geant4 VecGeom HepMC3)
  string(TOUPPER "${_dep}" _upper)
  celeritas_version_to_hex(_temp_version_hex ${_dep}_VERSION)
  list(APPEND CELERITAS_DEPENDENCY_VERSIONS
    "#define CELERITAS_${_upper}_VERSION ${_temp_version_hex}"
  )
endforeach()
string(JOIN "\n" CELERITAS_DEPENDENCY_VERSIONS ${CELERITAS_DEPENDENCY_VERSIONS})
celeritas_configure_file("celeritas_sys_config.h.in" "celeritas_sys_config.h" @ONLY)

#----------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------#

add_subdirectory(corecel)
add_subdirectory(geocel)
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
