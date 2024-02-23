#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
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
  instead of ``<package>``, and an ``@`` symbol can be used to find a specific
  version (e.g. ``Geant4@11.0``) which will be passed to the ``find_package``
  command.

.. command:: celeritas_set_default

  Set a locally-scoped value for the given variable if it is undefined.

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

.. command:: celeritas_install

  Install library that correctly deal with CUDA relocatable device code.

.. command:: celeritas_set_target_properties

  Install library that correctly deal with CUDA relocatable device code extra
  libraries.

.. command:: celeritas_add_library

  Add a library that correctly links against CUDA relocatable device code, has
  the ``Celeritas::`` aliases, and is generated into the ``lib/`` build
  directory.

.. command:: celeritas_add_object_library

  Add an OBJECT library to reduce dependencies (e.g. includes) from other libraries.

.. command:: celeritas_add_executable

  Create an executable and install it::

    celeritas_add_executable(<target> [<source> ...])

  The ``<target>`` is a unique identifier for the executable target. The actual
  executable name may end up with an .exe suffix (e.g. if it's windows). The
  executable will be built into the top-level ``bin`` directory, so all
  executables will sit side-by-side before installing.

  The ``<source>`` arguments are passed to CMake's builtin ``add_executable``
  command.

  .. command:: celeritas_target_link_libraries

  Specify libraries or flags to use when linking a given target and/or its dependents, taking
  in account the extra targets (see celeritas_rdc_add_library) needed to support CUDA relocatable
  device code.

    ::

      celeritas_target_link_libraries(<target>
        <PRIVATE|PUBLIC|INTERFACE> <item>...
        [<PRIVATE|PUBLIC|INTERFACE> <item>...]...))

  Usage requirements from linked library targets will be propagated to all four targets. Usage requirements
  of a target's dependencies affect compilation of its own sources. In the case that ``<target>`` does
  not contain CUDA code, the command decays to ``target_link_libraries``.

  See ``target_link_libraries`` for additional detail.


.. command:: celeritas_target_include_directories

  Add include directories to a target.

    ::

      celeritas_target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
        <INTERFACE|PUBLIC|PRIVATE> [items1...]
        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  Specifies include directories to use when compiling a given target. The named <target>
  must have been created by a command such as celeritas_rdc_add_library(), add_executable() or add_library(),
  and can be used with an ALIAS target. It is aware of the 4 underlying targets (objects, static,
  middle, final) present when the input target was created celeritas_rdc_add_library() and will propagate
  the include directories to all four. In the case that ``<target>`` does not contain CUDA code,
  the command decays to ``target_include_directories``.

  See ``target_include_directories`` for additional detail.

.. command:: celeritas_target_compile_options

   Specify compile options for a CUDA RDC target

     ::
       celeritas_target_compile_options(<target> [BEFORE]
         <INTERFACE|PUBLIC|PRIVATE> [items1...]
         [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  In the case that an input target does not contain CUDA code, the command decays
  to ``target_compile_options``.

  See ``target_compile_options`` for additional detail.

.. command:: celeritas_configure_file

  Configure to the build "include" directory for later installation::

    celeritas_configure_file(<input> <output> [ARGS...])

  The ``<input>`` must be a relative path to the current source directory, and
  the ``<output>` path is configured to the project build "include" directory.

.. comand:: celeritas_error_incompatible_option

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

.. command:: celeritas_version_to_hex

  Convert a version number to a C-formatted hexadecimal string::

    celeritas_version_to_hex(<var> <version_prefix>)

  The code will set a version to zero if ``<version_prefix>_VERSION`` is not
  found. If ``<version_prefix>_MAJOR`` is defined it will use that; otherwise
  it will try to split the version into major/minor/patch.

.. command:: celeritas_get_geant_libs

  Construct a list of Geant4 libraries for linking by adding a ``Geant4::G4``
  prefix and the necessary suffix::

    celeritas_get_g4libs(<var> [<lib> ...])

  Example::

    celeritas_get_g4libs(_g4libs geometry persistency)
    target_link_library(foo ${_g4libs})

  If Geant4 is unavailable, the result will be empty.

#]=======================================================================]
include_guard(GLOBAL)

include(CheckLanguage)
include(CudaRdcUtils)

set(CELERITAS_DEFAULT_VARIABLES)

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
    set(${name} "${value}")
  endif()
  list(APPEND CELERITAS_DEFAULT_VARIABLES ${name})
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
  cuda_rdc_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_rdc_sources_contains_cuda(_cuda_sources ${_sources})
  if(CELERITAS_USE_HIP AND _cuda_sources)
    # When building Celeritas libraries, we put HIP/CUDA files in shared .cu
    # suffixed files. Override the language if using HIP.
    set_source_files_properties(
      ${_cuda_sources}
      PROPERTIES LANGUAGE HIP
    )
  endif()

  celeritas_cuda_rdc_wrapper_add_library(${target} ${ARGN})

  # Add Celeritas:: namespace alias
  celeritas_cuda_rdc_wrapper_add_library(Celeritas::${target} ALIAS ${target})

  # Build all targets in lib/
  celeritas_set_target_properties(${target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
  )

  # We could hide this behind `if (CELERITAS_USE_ROOT)`
  get_target_property(_tgt ${target} CUDA_RDC_OBJECT_LIBRARY)
  if(_tgt)
    set_target_properties(${_tgt} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
    )
  endif()

  # Install all targets to lib/
  celeritas_install(TARGETS ${target}
    EXPORT celeritas-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_cuda_rdc_wrapper_add_library)
  if(NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_CUDA)
    add_library(${ARGV})
  else()
    cuda_rdc_add_library(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_set_target_properties)
  if(NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_CUDA)
    set_target_properties(${ARGV})
  else()
    cuda_rdc_set_target_properties(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_install)
  if(NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_CUDA)
    install(${ARGV})
  else()
    cuda_rdc_install(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_target_link_libraries)
  if(NOT CELERITAS_USE_VecGeom)
    target_link_libraries(${ARGV})
  else()
    cuda_rdc_target_link_libraries(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_target_include_directories)
  if(NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_CUDA)
    target_include_directories(${ARGV})
  else()
    cuda_rdc_target_include_directories(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_target_compile_options)
  if(NOT CELERITAS_USE_VecGeom OR NOT CELERITAS_USE_CUDA)
    target_compile_options(${ARGV})
  else()
    cuda_rdc_target_compile_options(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#

macro(celeritas_sources_contains_cuda)
  cuda_rdc_sources_contains_cuda(${ARGV})
endmacro()

#-----------------------------------------------------------------------------#
# Add an object library to limit the propagation of includes to the rest of the
# library.
function(celeritas_add_object_library target)
  add_library(${target} OBJECT ${ARGN})
  install(TARGETS ${target}
    EXPORT celeritas-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_add_executable target)
  add_executable("${target}" ${ARGN})
  install(TARGETS "${target}"
    EXPORT celeritas-targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT runtime
  )
  set_target_properties("${target}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CELERITAS_RUNTIME_OUTPUT_DIRECTORY}"
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

function(celeritas_error_incompatible_option  msg var new_value)
  message(SEND_ERROR "Invalid setting ${var}=${${var}}: ${msg}
    Possible fix: cmake -D${var}=${new_value} ${CMAKE_BINARY_DIR}"
  )
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

function(celeritas_version_to_hex var version)
  if(NOT DEFINED "${version}_MAJOR" AND DEFINED "${version}")
    # Split version into components
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _match "${${version}}")
    if(NOT _match)
      message(AUTHOR_WARNING
        "Failed to parse version string for ${version}=\"${${version}}\":
        _match=${_match}"
      )
    endif()
    set(${version}_MAJOR "${CMAKE_MATCH_1}")
    set(${version}_MINOR "${CMAKE_MATCH_2}")
    set(${version}_PATCH "${CMAKE_MATCH_3}")
  endif()
  # Set any empty or undefined values to zero
  foreach(_ext MAJOR MINOR PATCH)
    if(NOT ${version}_${_ext})
      set(${version}_${_ext} 0)
    endif()
  endforeach()
  # Add an extra 1 up front and strip it to zero-pad
  math(EXPR _temp_version
    "((256 + ${${version}_MAJOR}) * 256 + ${${version}_MINOR}) * 256 + ${${version}_PATCH}"
    OUTPUT_FORMAT HEXADECIMAL
  )
  string(SUBSTRING "${_temp_version}" 3 -1 _temp_version)
  set(${var} "0x${_temp_version}" PARENT_SCOPE)
endfunction()


#-----------------------------------------------------------------------------#

function(celeritas_get_g4libs var)
  set(_result)
  if(NOT CELERITAS_USE_Geant4)
    # No Geant4, no libs
  elseif(Geant4_VERSION VERSION_LESS 10.6)
    # Old Geant4 doesn't have granularity
    set(_result ${Geant4_LIBRARIES})
  else()
    # Determine the default library extension
    if(TARGET Geant4::G4global-static)
      set(_g4_static TRUE)
    else()
      set(_g4_static FALSE)
    endif()
    if(TARGET Geant4::G4global)
      set(_g4_shared TRUE)
    else()
      set(_g4_shared FALSE)
    endif()
    if(NOT _g4_static OR (BUILD_SHARED_LIBS AND _g4_shared))
      # Use shared if static is unavailable, or if shared is available and we're
      # building shared
      set(_ext "")
    else()
      set(_ext "-static")
    endif()

    foreach(_shortlib IN LISTS ARGN)
      set(_lib Geant4::G4${_shortlib}${_ext})
      if(TARGET "${_lib}")
        list(APPEND _result "${_lib}")
      elseif(_shortlib STREQUAL "persistency")
        # Workaround for differing target names in 11.2.0,1 for G4persistency
        # We do this here and not in FindGeant4 because we have no
        # guarantee projects using Celeritas and Geant4 won't mess with target
        # names themselves (and if we had to create a "celeritas::" target,
        # we'd still have to specialize here).
        list(APPEND _result Geant4::G4mctruth${_ext} Geant4::G4geomtext${_ext})
        if(TARGET "Geant4::G4gdml${_ext}")
          list(APPEND _result Geant4::G4gdml${_ext})
        endif()
      elseif(_shortlib STREQUAL "tasking")
        # Same workaround for tasking, which was split between G4global and G4run
        # from 11.1
        list(APPEND _result Geant4::G4run${_ext} Geant4::G4global${_ext})
      else()
        message(AUTHOR_WARNING "No Geant4 library '${_lib}' exists")
      endif()
    endforeach()
    # This avoids "ld: warning: ignoring duplicate libraries:"
    list(REMOVE_DUPLICATES _result)
  endif()
  set(${var} "${_result}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
