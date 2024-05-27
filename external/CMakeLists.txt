#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
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

#-----------------------------------------------------------------------------#
# GTest
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_GTest)
  FetchContent_Declare(
    GTest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  set(BUILD_GMOCK OFF) # For compatibility
  set(INSTALL_GTEST OFF)
  set(gtest_hide_internal_symbols ON)
  FetchContent_MakeAvailable(GTest)
endif()

#-----------------------------------------------------------------------------#
# nlohmann_json
#-----------------------------------------------------------------------------#

if(CELERITAS_EXTERNAL_nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  FetchContent_MakeAvailable(nlohmann_json)
endif()

#-----------------------------------------------------------------------------#
foreach(_var IN LISTS _checked_variables)
  set(_orig "${_saved_${_var}}")
  set(_new "${${_var}}")
  if(NOT "${_new}" STREQUAL "${_orig}")
    message(SEND_ERROR "An externally fetched package changed the value of ${_var} from ${_orig} to ${_new}. We are saving the original value as a cache variable: please reconfigure.")
    set(${_var} "${_orig}" CACHE STRING "Reset to Celeritas default" FORCE)
  endif()
endforeach()

#-----------------------------------------------------------------------------#
