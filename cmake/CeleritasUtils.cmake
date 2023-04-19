#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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

.. command:: celeritas_add_library

  Add a library that correctly links against CUDA relocatable device code, has
  the ``Celeritas::`` aliases, and is generated into the ``lib/`` build
  directory.

.. command:: celeritas_configure_file

  Configure to the build "include" directory for later installation::

    celeritas_configure_file(<input> <output> [ARGS...])

  The ``<input>`` must be a relative path to the current source directory, and
  the ``<output>` path is configured to the project build "include" directory.

.. command:: celeritas_setup_option

  Add a single compile time option value to the list::

    celeritas_setup_option(<var> <option> [conditional])

  This appends ``<option>`` to the list ``<var>_OPTIONS`` if ``${conditional}``
  is true or the argument is not provided; and to ``<var>_DISABLED_OPTIONS`` if
  the variable is present and false.

.. command:: celeritas_define_options

  Set up CMake variables for defining configure-time option switches::

    celeritas_define_options(<var> <doc>)

  This will set <var> to the first item of the list ``${<var>_OPTIONS}``. It
  will validate that the selection is one of the list and default to the first
  item in the list.

.. command:: celeritas_generate_option_macros

  Generate preprocessor macros for the given option list.::

    celeritas_generate_option_macros(<var>)

  This requires the list of ``<var>_OPTIONS`` to be set and ``<var>`` to be set,
  and it creates a string in the parent scope called ``<var>_MACROS``

  The resulting macro list starts a counter counter from 1 because undefined
  macros have the implicit value of 0 in the C preprocessor. Thus any
  unavailable options (e.g. CELERITAS_USE_CURAND when HIP is in use) will
  implicitly be zero.

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
    set(_reset_found OFF)
    list(GET _findpkg 0 _findpkg)
    if(NOT DEFINED ${_findpkg}_FOUND)
      find_package(${_findpkg} QUIET)
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

function(celeritas_add_library target)
  celeritas_rdc_add_library(${target} ${ARGN})

  # Add Celeritas:: namespace alias
  add_library(Celeritas::${target} ALIAS ${target})

  set(_targets ${target})
  get_target_property(_tgt ${target} CELERITAS_CUDA_FINAL_LIBRARY)
  if(_tgt)
    celeritas_strip_alias(_tgt ${_tgt})
    # Building with CUDA RDC support: add final library
    list(APPEND _targets ${_tgt})
    get_target_property(_tgt ${target} CELERITAS_CUDA_STATIC_LIBRARY)
    celeritas_strip_alias(_tgt ${_tgt})
    if(NOT _tgt STREQUAL target)
      # Shared and static library have different names
      list(APPEND _targets ${_tgt})
    endif()
    get_target_property(_tgt ${target} CELERITAS_CUDA_OBJECT_LIBRARY)
    if(_tgt)
      celeritas_strip_alias(_tgt ${_tgt})
      set_target_properties(${_tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()

  # Build all targets in lib/
  set_target_properties(${_targets} PROPERTIES ${_props}
    ARCHIVE_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
  )

  # Install all targets to lib/
  install(TARGETS ${_targets}
    EXPORT celeritas-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_configure_file input output)
  if(NOT IS_ABSOLUTE "${input}")
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/${input}")
  endif()
  configure_file("${input}"
    "${CELERITAS_HEADER_CONFIG_DIRECTORY}/${output}"
    ${ARGN})
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

  list(GET ${var}_OPTIONS 0 _default)
  set(${var} "${_default}" CACHE STRING "${doc}")
  set_property(CACHE ${var} PROPERTY STRINGS "${${var}_OPTIONS}")

  list(FIND ${var}_OPTIONS "${${var}}" _index)
  if(_index EQUAL -1)
    message(SEND_ERROR "Invalid value ${var}=${${var}}: must be "
      "${${var}_OPTIONS}; overriding for next configure")
    set(${var} "${_default}" CACHE STRING "${doc}" FORCE)
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_generate_option_macros var)
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
  string(JOIN "\n" _result
    ${_options}
    "#define ${var} ${var}_${_val}"
  )

  # Set in parent scope
  set(${var}_MACROS "${_result}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
