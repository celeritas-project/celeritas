#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindLArSoft
--------------

Find LArSoft data object model library. This requires
additional fermilab-related helpers to provide the necessary ``FindX.cmake``
modules.

TODO: when LArSoft switches to Phlex, make ``art`` or ``phlex`` a COMPONENT,
and search for dependencies based on the installed version of ``cetmodules``.

#]=======================================================================]
if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_larsoft_quiet QUIET)
else()
  set(_larsoft_quiet)
endif()

# Suppress dualing Boost find messages
set(Boost_FIND_QUIETLY TRUE)

# Ordered dependencies
set(_required_vars)
foreach(_module cetmodules art art_root_io larcore lardataobj)
  list(APPEND _required_vars ${_module}_DIR)
  if(NOT ${_module}_FOUND)
    find_package(${_module} ${_larsoft_quiet})
  endif()
endforeach()

list(REVERSE _required_vars)

# Compute a fingerprint from the resolved package directory values so we can
# detect when the environment has changed and the cached version is stale.
set(_larsoft_fingerprint "")
foreach(_var IN LISTS _required_vars)
  string(APPEND _larsoft_fingerprint "|${${_var}}")
endforeach()

# Cache the version string and invalidate it whenever the package dirs change.
if(NOT "${_larsoft_fingerprint}" STREQUAL "${_LArSoft_dirs_fingerprint}")
  # Read and filter LARSOFT_VERSION from the environment.
  # Converts the UPS-style tag (e.g. v10_14_01) to a dotted version (10.14.1).
  if(DEFINED ENV{LARSOFT_VERSION})
    set(_larsoft_env_version "$ENV{LARSOFT_VERSION}")
    string(REGEX REPLACE "^[Vv]" "" _larsoft_env_version "${_larsoft_env_version}")
    string(REPLACE "_" "." _larsoft_env_version "${_larsoft_env_version}")
  else()
    set(_larsoft_env_version "")
  endif()

  set(_LArSoft_dirs_fingerprint "${_larsoft_fingerprint}"
    CACHE INTERNAL "Fingerprint of LArSoft package dirs for version cache invalidation")
  set(LArSoft_VERSION "${_larsoft_env_version}"
    CACHE INTERNAL "Filtered LArSoft version from LARSOFT_VERSION environment variable")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME}
  VERSION_VAR LArSoft_VERSION
  REQUIRED_VARS ${_required_vars}
)
unset(_larsoft_quiet)
unset(_module)
unset(_required_vars)
unset(_larsoft_fingerprint)
unset(_larsoft_env_version)

#-----------------------------------------------------------------------------#
