#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasOptionUtils
--------------------

CMake configuration utility functions for Celeritas, primarily for option setup.

.. command:: celeritas_find_or_builtin_package

  Look for a *required* external dependency ``<package>`` and cache whether we
  found it or not.

    celeritas_find_or_builtin_package(<package> [...])

  The cache variable ``CELERITAS_BUILTIN_<package>`` is used so that on
  subsequent configures we do not "find" an external that we
  configured/installed ourself as a built-in (since CMake's search path includes
  the installation prefix). Additional arguments (e.g. a version number) will be
  forwarded to ``find_package``.

  Previous logic must determine whether the package is needed or not: if not
  already found, this will enable the builtin copy for later use.

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
  instead of ``<package>``, and an ``@`` symbol can be used to find a specific
  version (e.g. ``Geant4@11.0``) which will be passed to the ``find_package``
  command.

.. command:: celeritas_set_default

  Set a value for the given variable if it is undefined. If a
  third argument is given and Celeritas is the top-level project, create a cache
  variable with the given documentation; otherwise, set the variable locally
  so it's scoped only to Celeritas. ::

     celeritas_set_default(<variable> <value>)
     celeritas_set_default(<variable> <value> <doc>)
     celeritas_set_default(<variable> <value> <type> <doc>)

.. command:: celeritas_check_python_module

   Determine whether a given Python module is available with the current
   environment. ::

     celeritas_check_python_module(<variable> <module>)

   ``<variable>``
     Variable name that will be set to whether the module exists

   ``<module>``
     Python module name, e.g. "numpy" or "scipy.linalg"

   Note that because this function caches the Python script result to save
   reconfigure time (or when multiple scripts check for the same module),
   changing the Python executable or installed modules may mean
   having to delete or modify your CMakeCache.txt file.

   Example::

      celeritas_check_python_module(has_numpy "numpy")

.. command:: celeritas_error_incompatible_option

  Print a descriptive failure message about conflicting cmake options.

    celeritas_error_incompatible_option(<msg> <var> <conflict_var> <new_value>)

  Here the recommendation will be to change conflict_var to new_value.

.. command:: celeritas_setup_option

  Add a single compile time option value to the list::

    celeritas_setup_option(<var> <option> [conditional])

  This appends ``<option>`` to the list ``<var>_OPTIONS`` if ``${conditional}``
  is true or the argument is not provided; and to ``<var>_DISABLED_OPTIONS`` if
  the variable is present and false.

.. command:: celeritas_define_options

  Set up CMake variables for defining configure-time option switches::

    celeritas_define_options(<var> <doc>)

  If <var> is not yet set, this will set it to the first item of the list
  ``${<var>_OPTIONS}`` *without* storing it in the cache, so that it will be
  set locally but will not persist if other CMake options change.  If provided
  by the user, this command will validate that the pre-existing selection is one
  of the list.

.. command:: celeritas_generate_option_config

  Generate ``#define`` macros for the given option list.::

    celeritas_generate_option_config(<var>)

  This requires the list of ``<var>_OPTIONS`` to be set and ``<var>`` to be set,
  and it creates a string in the parent scope called ``<var>_CONFIG`` for use in
  a configure file (such as ``celeritas_config.h``).

  The resulting macro list starts its counter 1 because undefined
  macros have the implicit value of 0 in the C preprocessor. Thus any
  unavailable options (e.g. CELERITAS_USE_CURAND when HIP is in use) will
  implicitly be zero.

#]=======================================================================]
include_guard(GLOBAL)

include(CheckLanguage)

# List of variables configured via `celeritas_set_default`
set(CELERITAS_DEFAULT_VARIABLES)
# True if any CELERITAS_BUILTIN_XXX
set(CELERITAS_BUILTIN FALSE)

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
macro(celeritas_find_or_builtin_package package)
  set(_var "CELERITAS_BUILTIN_${package}")
  set(_use_builtin "${${_var}}")
  if(NOT _use_builtin)
    if(NOT ${package}_FOUND)
      # Only look for the package if it's not "found" (which can occur if
      # Celeritas is a subproject)
      find_package(${package} ${ARGN})
    endif()
    set(_found "${${package}_FOUND}")
    if(NOT DEFINED ${_var})
      if(NOT _found)
        set(_use_builtin ON)
      endif()
      set("${_var}" "${_use_builtin}" CACHE BOOL
        "Fetch and build ${package}")
      mark_as_advanced(${_var})
    elseif(NOT _found)
      # First time looking, builtin defined, package not fuond
      find_package(${package} ${ARGN} QUIET REQUIRED)
    endif()
    unset(_found)
  endif()
  if(_use_builtin)
    set(CELERITAS_BUILTIN TRUE)
  endif()
  unset(_use_builtin)
  unset(_var)
endmacro()

#-----------------------------------------------------------------------------#

# Note: this is a macro so that `find_package` variables stay in the global
# scope.
macro(celeritas_optional_package package)
  if("${ARGC}" EQUAL 2)
    set(_findpkg "${package}")
    set(_findversion)
    set(_docstring "${ARGV1}")
  else()
    set(_findpkg "${ARGV1}")
    set(_findversion)
    if(_findpkg MATCHES "([^@]+)@([^@]+)")
      set(_findpkg ${CMAKE_MATCH_1})
      set(_findversion ${CMAKE_MATCH_2})
    endif()
    set(_docstring "${ARGV2}")
  endif()

  set(_var "CELERITAS_USE_${package}")
  if(DEFINED "${_var}")
    set(_val "${_var}")
  else()
    set(_reset_found OFF)
    list(GET _findpkg 0 _findpkg)
    if(NOT DEFINED ${_findpkg}_FOUND)
      find_package(${_findpkg} ${_findversion} QUIET)
      set(_reset_found ON)
    endif()
    celeritas_to_onoff(_val ${${_findpkg}_FOUND})
    message(STATUS "Set ${_var}=${_val} based on package availability")
    if(_reset_found)
      unset(${_findpkg}_FOUND)
    endif()
  endif()

  option("${_var}" "${_docstring}" "${_val}")
endmacro()

#-----------------------------------------------------------------------------#

macro(celeritas_set_default name value)
  if(NOT DEFINED ${name})
    message(VERBOSE "Celeritas: set default ${name}=${value}")
    if(PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME AND "${ARGC}" GREATER 2)
      if("${ARGC}" EQUAL 3)
        option(${name} "${ARGV2}" "${value}")
      else()
        set(${name} "${value}" CACHE ${ARGN})
      endif()
    else()
      set(${name} "${value}")
      list(APPEND CELERITAS_DEFAULT_VARIABLES ${name})
    endif()
  endif()
endmacro()

#-----------------------------------------------------------------------------#

function(celeritas_check_python_module varname module)
  set(_cache_name CELERITAS_CHECK_PYTHON_MODULE_${module})
  if(DEFINED ${_cache_name})
    # We've already checked for this module
    set(_found "${${_cache_name}}")
  else()
    message(STATUS "Check Python module ${module}")
    set(_cmd
      "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CELERITAS_PYTHONPATH}"
      "${Python_EXECUTABLE}" -c "import ${module}"
    )
    execute_process(COMMAND
      ${_cmd}
      RESULT_VARIABLE _result
      ERROR_QUIET # hide error message if module unavailable
    )
    # Note: use JSON-compatible T/F representation
    if(_result)
      set(_msg "not found")
      set(_found false)
    else()
      set(_msg "found")
      set(_found true)
    endif()
    message(STATUS "Check Python module ${module} -- ${_msg}")
    set(${_cache_name} "${_found}" CACHE INTERNAL
      "Whether Python module ${module} is available")
  endif()

  # Save outgoing variable
  set(${varname} "${_found}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_error_incompatible_option  msg var new_value)
  message(SEND_ERROR "Invalid setting ${var}=${${var}}: ${msg}")
  message(WARNING "Setting ${var}=${new_value} for next build")
  set(${var} "${new_value}" CACHE STRING "Set automatically: ${msg}" FORCE)
endfunction()

#-----------------------------------------------------------------------------#

macro(celeritas_setup_option var option) #[condition]
  if(${ARGC} EQUAL 2)
    # always-on-option
    list(APPEND ${var}_OPTIONS ${option})
  elseif(${ARGV2})
    # variable evaluates to true
    list(APPEND ${var}_OPTIONS ${option})
  else()
    list(APPEND ${var}_DISABLED_OPTIONS ${option})
  endif()
endmacro()

#-----------------------------------------------------------------------------#

function(celeritas_define_options var doc)
  if(NOT ${var}_OPTIONS)
    message(FATAL_ERROR "${var}_OPTIONS has no options")
  endif()
  mark_as_advanced(${var}_OPTIONS)

  # Current value and var name for previous value
  set(_val "${${var}}")
  set(_last_var _LAST_${var})

  # Add variable to cache with current value if given (empty is default)
  set(${var} "${_val}" CACHE STRING "${doc}")
  set_property(CACHE ${var} PROPERTY STRINGS "${${var}_OPTIONS}")

  if(_val STREQUAL "")
    # Dynamic default option: set as core variable in parent scope
    list(GET ${var}_OPTIONS 0 _default)
    set(${var} "${_default}" PARENT_SCOPE)
    set(_val "${_default}")
    if(NOT "${_val}" STREQUAL "${${_last_var}}")
      message(STATUS "Set ${var}=${_val}")
    endif()
  else()
    # User-provided value: check against list
    list(FIND ${var}_OPTIONS "${_val}" _index)
    if(_index EQUAL -1)
      string(JOIN "," _optlist ${${var}_OPTIONS})
      celeritas_error_incompatible_option(
        "valid options are {${_optlist}} "
        "${var}" "${_default}"
      )
    endif()
  endif()
  set(${_last_var} "${_val}" CACHE INTERNAL "")
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_generate_option_config var)
  # Add disabled options first
  set(_options)
  foreach(_val IN LISTS ${var}_DISABLED_OPTIONS)
    string(TOUPPER "${_val}" _val)
    list(APPEND _options "#define ${var}_${_val} 0")
  endforeach()

  # Add available options
  set(_counter 1)
  foreach(_val IN LISTS ${var}_OPTIONS)
    string(TOUPPER "${_val}" _val)
    list(APPEND _options "#define ${var}_${_val} ${_counter}")
    math(EXPR _counter "${_counter} + 1")
  endforeach()

  # Add selected option
  string(TOUPPER "${${var}}" _val)
  if(NOT _val)
    message(FATAL_ERROR "Option configuration '${var}' is undefined")
  endif()
  string(JOIN "\n" _result
    ${_options}
    "#define ${var} ${var}_${_val}"
  )

  # Set in parent scope
  set(${var}_CONFIG "${_result}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
