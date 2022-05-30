#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasUtils
--------------

CMake configuration utility functions for Celeritas.

.. command:: celeritas_optional_language

  Add an configurable cache option ``CELERITAS_USE_<lang>`` that defaults to
  checking whether the language is available.

    celeritas_optional_language(<lang>)

.. command:: celeritas_optional_package

  Add an configurable cache option ``CELERITAS_USE_<package>`` that searches for
  the package to decide its default value.

    celeritas_optional_package(<package> [<find_package>] <docstring>)

  This won't be used for all Celeritas options or even all external dependent
  packages. If given, the ``<find_package>`` package name will searched for
  instead of ``<package>``.

#]=======================================================================]
include_guard(GLOBAL)

include(CheckLanguage)

#-----------------------------------------------------------------------------#

function(celeritas_optional_language lang)
  set(_var "CELERITAS_USE_${lang}")
  if(DEFINED "${_var}")
    set(_val "${_var}")
  else()
    check_language(${lang})
    set(_val OFF)
    if(CMAKE_${lang}_COMPILER)
      set(_val ON)
    endif()
    message(STATUS "Set ${_var}=${_val} based on compiler availability")
  endif()

  option("${_var}" "Enable the ${lang} language" "${_val}" )
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_to_onoff varname)
  if(ARGC GREATER 1 AND ARGV1)
    set(${varname} ON PARENT_SCOPE)
  else()
    set(${varname} OFF PARENT_SCOPE)
  endif()
endfunction()

#-----------------------------------------------------------------------------#

# Note: this is a macro so that `find_package` variables stay in the global
# scope.
macro(celeritas_optional_package package)
  if("${ARGC}" EQUAL 2)
    set(_findpkg "${package}")
    set(_docstring "${ARGV1}")
  else()
    set(_findpkg "${ARGV1}")
    set(_docstring "${ARGV2}")
  endif()

  set(_var "CELERITAS_USE_${package}")
  if(DEFINED "${_var}")
    set(_val "${_var}")
  else()
    find_package(${_findpkg} QUIET)
    list(GET _findpkg 0 _findpkg)
    celeritas_to_onoff(_val ${${_findpkg}_FOUND})
    message(STATUS "Set ${_var}=${_val} based on package availability")
  endif()

  option("${_var}" "${_docstring}" "${_val}")
endmacro()

#-----------------------------------------------------------------------------#
