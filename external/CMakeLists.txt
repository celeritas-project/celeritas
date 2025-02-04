#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(_celer_external_packages)

#-----------------------------------------------------------------------------#
# GTest
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_GTest)
  FetchContent_Declare(
    GTest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  list(APPEND _celer_external_packages GTest)

  set(BUILD_GMOCK OFF) # For compatibility
  set(INSTALL_GTEST OFF)
  set(gtest_hide_internal_symbols ON)
  set(gtest_force_shared_crt ON) # Don't replace /MD with -MT for windows
endif()

#-----------------------------------------------------------------------------#
# nlohmann_json
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  list(APPEND _celer_external_packages nlohmann_json)

  set(JSON_Install ON)
endif()

#-----------------------------------------------------------------------------#
# Perfetto
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_Perfetto)
  if(PERFETTO_ROOT AND NOT FETCHCONTENT_SOURCE_DIR_PERFETTO)
    set(FETCHCONTENT_SOURCE_DIR_PERFETTO ${PERFETTO_ROOT})
  endif()
  FetchContent_Declare(
    Perfetto
    URL https://github.com/google/perfetto/archive/refs/tags/v49.0.tar.gz
  )
  list(APPEND _celer_external_packages Perfetto)
endif()

#-----------------------------------------------------------------------------#
# G4VG
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_G4VG)
  FetchContent_Declare(
    G4VG
    URL https://github.com/celeritas-project/g4vg/releases/download/v1.0.1/g4vg-1.0.1.tar.gz
  )
  list(APPEND _celer_external_packages G4VG)
  foreach(_var BUILD_TESTS DEBUG)
    set(G4VG_${_var} CELERITAS_${_var})
  endforeach()
endif()

#-----------------------------------------------------------------------------#
# BUILD
#-----------------------------------------------------------------------------#

# Save some variables to see if badly behaving packages change them
set(_checked_cmake_variables
  BUILD_SHARED_LIBS CMAKE_CXX_STANDARD CMAKE_CXX_EXTENSIONS
)
if(NOT WIN32)
  # RPath doesn't matter on windows, and some CI externalas change them
  list(APPEND _checked_cmake_variables
    CMAKE_INSTALL_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
  )
endif()
foreach(_var IN LISTS _checked_cmake_variables)
  set(_saved_${_var} "${${_var}}")
endforeach()

# Activate externals
FetchContent_MakeAvailable(${_celer_external_packages})

# Check that variables are unmodified
foreach(_var IN LISTS _checked_cmake_variables)
  set(_orig "${_saved_${_var}}")
  set(_new "${${_var}}")
  if(NOT "${_new}" STREQUAL "${_orig}")
    message(SEND_ERROR "An externally fetched package changed the value of ${_var} from ${_orig} to ${_new}. We are saving the original value as a cache variable: please reconfigure.")
    set(${_var} "${_orig}" CACHE STRING "Reset to Celeritas default" FORCE)
  endif()
endforeach()

#-----------------------------------------------------------------------------#
# POST-FETCH CONFIG
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_Perfetto AND perfetto_POPULATED)
  set(Perfetto_INCLUDE_DIR ${perfetto_SOURCE_DIR}/sdk)
  add_library(celeritas_perfetto STATIC ${Perfetto_INCLUDE_DIR}/perfetto.cc)
  add_library(Celeritas::Perfetto ALIAS celeritas_perfetto)
  target_include_directories(celeritas_perfetto INTERFACE ${Perfetto_INCLUDE_DIR})
endif()

#-----------------------------------------------------------------------------#
