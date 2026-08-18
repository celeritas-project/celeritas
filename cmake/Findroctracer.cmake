#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findroctracer
-------------

Find the roctracer library.

#]=======================================================================]

set(_hints "$ENV{ROCM_PATH}" "${ROCM_PATH}" "/opt/rocm")

find_library(ROCTX_LIBRARY
  NAMES roctx64 roctx
  HINTS ${_hints}
  PATH_SUFFIXES lib lib64
)
mark_as_advanced(ROCTX_LIBRARY)
set(roctracer_LIBRARY "${ROCTX_LIBRARY}")
set(roctracer_LIBRARIES "${ROCTX_LIBRARY}")

find_path(ROCTRACER_INCLUDE_DIR
  "roctracer/roctracer.h"
  HINTS ${_hints}
  PATH_SUFFIXES include
)
mark_as_advanced(ROCTRACER_INCLUDE_DIR)
set(roctracer_INCLUDE_DIR "${ROCTRACER_INCLUDE_DIR}")
set(roctracer_INCLUDE_DIRS "${ROCTRACER_INCLUDE_DIR}")

if(ROCTRACER_INCLUDE_DIR)
  set(_vers_regex "ROCTRACER_VERSION_(MAJOR|MINOR)[ 	]+([0-9]+)")
  file(STRINGS "${ROCTRACER_INCLUDE_DIR}/roctracer/roctracer.h"
    _roctracer_version_lines
    REGEX "${_vers_regex}"
  )
  foreach(_line IN LISTS _roctracer_version_lines)
    string(REGEX MATCH "${_vers_regex}" _line "${_line}")
    set(roctracer_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endforeach()
  string(JOIN "." roctracer_VERSION ${roctracer_VERSION_MAJOR} ${roctracer_VERSION_MINOR})
  unset(_vers_regex)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roctracer
  REQUIRED_VARS ROCTX_LIBRARY ROCTRACER_INCLUDE_DIR
  VERSION_VAR roctracer_VERSION
)

if(roctracer_FOUND AND NOT TARGET roctracer::roctx)
  add_library(roctracer::roctx INTERFACE IMPORTED)
  set_target_properties(roctracer::roctx PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCTRACER_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${ROCTX_LIBRARY}"
  )
endif()

unset(_hints)
unset(_line)
unset(_roctracer_version_lines)

#-----------------------------------------------------------------------------#
