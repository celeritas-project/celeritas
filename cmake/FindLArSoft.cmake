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

# cetmodules is lowest dependency
if(NOT cetmodules_FOUND)
  find_package(cetmodules ${_larsoft_quiet})
endif()
if(NOT art_FOUND)
  find_package(art ${_larsoft_quiet})
endif()
if(NOT lardataobj_FOUND)
  find_package(lardataobj ${_larsoft_quiet})
endif()
if(lardataobj_FOUND)
  set(LArSoft_VERSION ${lardataobj_VERSION})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME}
  REQUIRED_VARS lardataobj_DIR art_DIR cetmodules_DIR
)
unset(_larsoft_quiet)

#-----------------------------------------------------------------------------#
