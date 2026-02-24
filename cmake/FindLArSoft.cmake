#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindLArSoft
--------------

Find LArSoft data object model library. This requires
additional fermilab-related helpers to provide the necessary ``FindX.cmake``
modules.

#]=======================================================================]
if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_larsoft_quiet QUIET)
else()
  set(_larsoft_quiet)
endif()

# Ordered dependencies
set(_required_vars)
foreach(_module cetmodules art art_root_io larcore lardataobj)
  list(APPEND _required_vars ${_module}_DIR)
  if(NOT ${_module}_FOUND)
    find_package(${_module} ${_larsoft_quiet})
  endif()
endforeach()

# Set version from lardataobj
if(lardataobj_FOUND)
  set(LArSoft_VERSION ${lardataobj_VERSION})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS ${_required_vars}
)
unset(_larsoft_quiet)

#-----------------------------------------------------------------------------#
