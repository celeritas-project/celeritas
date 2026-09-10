#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindLArSoft
--------------

Find LArSoft dependencies. A set of FNAL-proivided FindX modules that live
in "cetmodules" *must* be found before larsim or other components
are searched for.

TODO: when LArSoft switches to Phlex, make ``art`` or ``phlex`` a COMPONENT,
and search for dependencies based on the installed version of ``cetmodules``.

#]=======================================================================]
include(CMakeFindDependencyMacro)

# Suppress dueling Boost find messages
set(Boost_FIND_QUIETLY TRUE)

# Ordered dependencies
set(_required_vars)
foreach(_module cetmodules art art_root_io larcore lardataobj larsim)
  list(APPEND _required_vars ${_module}_DIR)
  if(NOT ${_module}_FOUND)
    find_dependency(${_module})
  endif()
endforeach()
set(LArSoft_VERSION ${larsim_VERSION})

list(REVERSE _required_vars)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME}
  VERSION_VAR LArSoft_VERSION
  REQUIRED_VARS ${_required_vars}
)
unset(_larsoft_quiet)
unset(_module)
unset(_required_vars)

#-----------------------------------------------------------------------------#
