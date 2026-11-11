#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findhipcub
--------

Find the hipcub library.

#]=======================================================================]

find_package(hipcub QUIET CONFIG HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}")

include(FindPackageHandleStandardArgs)

if(hipcub_FOUND)
  find_package_handle_standard_args(hipcub CONFIG_MODE)
else()
  # Fall back to searching for the version header
  find_path(hipcub_INCLUDE_DIR
    NAMES hipcub/hipcub_version.hpp
    HINTS "${CMAKE_HIP_COMPILER_ROCM_ROOT}"
    PATH_SUFFIXES include
  )
  if(hipcub_INCLUDE_DIR)
    set(_hipcub_version_hpp "${hipcub_INCLUDE_DIR}/hipcub/hipcub_version.hpp")
    file(STRINGS "${_hipcub_version_hpp}" _hipcub_version_lines
      REGEX "#define[ \t]+HIPCUB_VERSION_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
    foreach(_comp MAJOR MINOR PATCH)
      set(_hipcub_${_comp} 0)
      foreach(_line IN LISTS _hipcub_version_lines)
        if(_line MATCHES "HIPCUB_VERSION_${_comp}[ \t]+([0-9]+)")
          set(_hipcub_${_comp} "${CMAKE_MATCH_1}")
        endif()
      endforeach()
    endforeach()
    set(hipcub_VERSION
      "${_hipcub_MAJOR}.${_hipcub_MINOR}.${_hipcub_PATCH}")
    unset(_hipcub_version_lines)
  endif()
  find_package_handle_standard_args(hipcub
    REQUIRED_VARS hipcub_INCLUDE_DIR
    VERSION_VAR hipcub_VERSION
  )
  if(hipcub_FOUND AND NOT TARGET hip::hipcub)
    add_library(hip::hipcub INTERFACE IMPORTED)
    set_target_properties(hip::hipcub PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${hipcub_INCLUDE_DIR}")
  endif()
endif()

#-----------------------------------------------------------------------------#
