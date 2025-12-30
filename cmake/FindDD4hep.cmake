#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindDD4hep
----------

Find the Detector Description Toolkit for High Energy Physics.

#]=======================================================================]

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  cmake_language(GET_MESSAGE_LOG_LEVEL _orig_log_lev)
else()
  set(_orig_log_lev ${CMAKE_MESSAGE_LOG_LEVEL})
endif()

if(NOT _orig_log_lev OR _orig_log_lev STREQUAL "STATUS")
  # Suppress verbose DD4hep cmake configuration output
  set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
endif()

find_package(DD4hep QUIET CONFIG)

set(CMAKE_MESSAGE_LOG_LEVEL ${_orig_log_lev})
unset(_orig_log_lev)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DD4hep CONFIG_MODE)

#-----------------------------------------------------------------------------#
