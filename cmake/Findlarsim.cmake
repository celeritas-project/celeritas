#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findlarsim
----------

Find CMake files to use as a module with LArSim.

#]=======================================================================]

if(cetmodules_FOUND)
  find_package(${CMAKE_FIND_PACKAGE_NAME} QUIET CONFIG)
elseif(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  message(STATUS "Cannot search for ${CMAKE_FIND_PACKAGE_NAME}: cetmodules not found")
endif()
include(FindPackageHandleStandardArgs)
set(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY OFF)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME} CONFIG_MODE)

#-----------------------------------------------------------------------------#
