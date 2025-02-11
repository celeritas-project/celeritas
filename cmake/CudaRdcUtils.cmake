#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors, for more details see:
#   https://github.com/celeritas-project/celeritas/blob/develop/COPYRIGHT
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CudaRdcUtils
------------

CMake utility functions for building and linking libraries containing CUDA
relocatable device code and most importantly linking against those libraries.

.. command:: cuda_rdc_add_library

  Add a library to the project using the specified source files *with* special
  handling for the case where the library contains CUDA relocatable device code.

  ::

    cuda_rdc_add_library(<name> [STATIC | SHARED | MODULE | ALIAS]
            [EXCLUDE_FROM_ALL]
            [<source>...])

  To support CUDA relocatable device code, the following 4 targets will be
  constructed:

  - A object library used to compile the source code and share the result with
    the static and shared library
  - A static library used as input to ``nvcc -dlink``
  - A shared “intermediary” library containing all the ``.o`` files but NO ``nvcc -dlink`` result
  - A shared “final” library containing the result of ``nvcc -dlink`` and linked against the "intermediary" shared library.

  An executable needs to load exactly one result of ``nvcc -dlink`` whose input
  needs to be the ``.o`` files from all the CUDA libraries it uses/depends-on.
  So if the executable has CUDA code, it will call ``nvcc -dlink`` itself and
  link against the "intermediary" shared libraries.  If the executable has no
  CUDA code, then it needs to link against the "final" library (of its most
  derived dependency). If the executable has no CUDA code but uses more than one
  CUDA library, it will still need to run its own ``nvcc -dlink`` step.


.. command:: cuda_rdc_target_link_libraries

  Specify libraries or flags to use when linking a given target and/or its
  dependents, taking in account the extra targets (see cuda_rdc_add_library)
  needed to support CUDA relocatable device code.

    ::

      cuda_rdc_target_link_libraries(<target>
        <PRIVATE|PUBLIC|INTERFACE> <item>...
        [<PRIVATE|PUBLIC|INTERFACE> <item>...]...))

  Usage requirements from linked library targets will be propagated to all four
  targets. Usage requirements of a target's dependencies affect compilation of
  its own sources. In the case that ``<target>`` does not contain CUDA code, the
  command decays to ``target_link_libraries``.

  See ``target_link_libraries`` for additional detail.


.. command:: cuda_rdc_target_include_directories

  Add include directories to a target.

    ::

      cuda_rdc_target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
        <INTERFACE|PUBLIC|PRIVATE> [items1...]
        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  Specifies include directories to use when compiling a given target. The named
  <target> must have been created by a command such as cuda_rdc_add_library(),
  add_executable() or add_library(), and can be used with an ALIAS target. It is
  aware of the 4 underlying targets (objects, static, middle, final) present
  when the input target was created cuda_rdc_add_library() and will propagate
  the include directories to all four. In the case that ``<target>`` does not
  contain CUDA code, the command decays to ``target_include_directories``.

  See ``target_include_directories`` for additional detail.


.. command:: cuda_rdc_install

  Specify installation rules for a CUDA RDC target.

    ::
      cuda_rdc_install(TARGETS targets... <ARGN>)

  In the case that an input target does not contain CUDA code, the command decays
  to ``install``.

  See ``install`` for additional detail.

.. command:: cuda_rdc_target_compile_options

   Specify compile options for a CUDA RDC target

     ::
       cuda_rdc_target_compile_options(<target> [BEFORE]
         <INTERFACE|PUBLIC|PRIVATE> [items1...]
         [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  In the case that an input target does not contain CUDA code, the command decays
  to ``target_compile_options``.

  See ``target_compile_options`` for additional detail.

.. command:: cuda_rdc_set_target_properties

    Set the targets properties.

     ::
       cuda_rdc_set_target_properties(<targets> ...
                      PROPERTIES <prop1> <value1>
                      [<prop2> <value2>] ...)

  In the case that an input target does not contain CUDA code, the command decays
  to ``set_target_properties``.

  See ``set_target_properties`` for additional detail.

#]=======================================================================]

include_guard(GLOBAL)

cmake_policy(VERSION 3.24...3.31)
# 3.19 is needed for set properties in INTERFACE libraries.
# 3.24 is needed to be able to mark the final library as always `-Wl,-no-as-needed`

#-----------------------------------------------------------------------------#

define_property(TARGET PROPERTY CUDA_RDC_LIBRARY_TYPE
  BRIEF_DOCS "Indicate the type of cuda library (STATIC and SHARED for nvlink usage, FINAL for linking into not cuda library/executable"
  FULL_DOCS "Indicate the type of cuda library (STATIC and SHARED for nvlink usage, FINAL for linking into not cuda library/executable"
)
define_property(TARGET PROPERTY CUDA_RDC_FINAL_LIBRARY
  BRIEF_DOCS "Name of the final library corresponding to this cuda library"
  FULL_DOCS "Name of the final library corresponding to this cuda library"
)
define_property(TARGET PROPERTY CUDA_RDC_STATIC_LIBRARY
  BRIEF_DOCS "Name of the static library corresponding to this cuda library"
  FULL_DOCS "Name of the static library corresponding to this cuda library"
)
define_property(TARGET PROPERTY CUDA_RDC_MIDDLE_LIBRARY
  BRIEF_DOCS "Name of the shared (without nvlink step) library corresponding to this cuda library"
  FULL_DOCS "Name of the shared (without nvlink step) library corresponding to this cuda library"
)
define_property(TARGET PROPERTY CUDA_RDC_OBJECT_LIBRARY
  BRIEF_DOCS "Name of the object (without nvlink step) library corresponding to this cuda library"
  FULL_DOCS "Name of the object (without nvlink step) library corresponding to this cuda library"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # This is necessary on platform that default to calling ld with --as-needed
  # (we are looking at you Ubuntu) but also in case user as explicitly using it.

  # This is used for the final libraries because of the following:
  #   * userlib_with_rdc.so has undefined symbols found in userlib_with_rdc_final (the result of nvlink)
  #   * userlib_with_rdc_final.so requires the userlib_with_rdc’s object files as input to nvlink
  #   * userlib_with_rdc_final.so also requires userlib_with_rdc.so for user's code.
  #   * user executable requires userlib_with_rdc.so due to direct dependencies.
  # So the link line order to produce the user executable as to be:
  #   .... userlib_with_rdc_final.so userlib_with_rdc.so
  # But userlib_with_rdc_final.so does not contains any symbol that is explicit used in the
  # user executable.  Consequently, if the linker is in the `--as-needed` mode it will drop
  # userlib_with_rdc_final.so.  However, it will then not be able to be used to resolve
  # the missing symbol needed by userlib_with_rdc.so

  # This implementation requires CMake version 3.24 or higher.  For more details see:
  #   https://cmake.org/cmake/help/latest/variable/CMAKE_LINK_LIBRARY_USING_FEATURE.html
  set(CMAKE_CXX_LINK_LIBRARY_USING_rdc_no_as_needed_SUPPORTED TRUE CACHE INTERNAL "")
  set(CMAKE_CXX_LINK_LIBRARY_USING_rdc_no_as_needed
    "LINKER:--push-state,--no-as-needed"
    "<LINK_ITEM>"
    "LINKER:--pop-state"
    CACHE INTERNAL ""
  )
else()
  set(CMAKE_CXX_LINK_LIBRARY_USING_no_as_needed_SUPPORTED FALSE)
endif()

##############################################################################
# Separate the OPTIONS out from the sources
#
macro(cuda_rdc_get_sources_and_options _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if(arg STREQUAL "OPTIONS")
      set( _found_options TRUE )
    elseif(
        arg STREQUAL "WIN32" OR
        arg STREQUAL "MACOSX_BUNDLE" OR
        arg STREQUAL "EXCLUDE_FROM_ALL" OR
        arg STREQUAL "STATIC" OR
        arg STREQUAL "SHARED" OR
        arg STREQUAL "MODULE" OR
        arg STREQUAL "ALIAS"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()


#-----------------------------------------------------------------------------#
# cuda_rdc_set_properties
#
# Set the property on a library and all its Cuda RDC support libraries.
#
function(cuda_rdc_set_target_properties target)
  cuda_rdc_strip_alias(target ${target})
  get_target_property(_targettype ${target} CUDA_RDC_LIBRARY_TYPE)
  if (NOT _targettype)
    set_target_properties(${target} ${ARGN})
    return()
  endif()
  foreach(_type "FINAL" "MIDDLE" "STATIC" "OBJECT")
    get_target_property(_lib ${target} "CUDA_RDC_${_type}_LIBRARY")
    if(_lib)
      cuda_rdc_strip_alias(_lib ${_lib})
      set_target_properties(${_lib} ${ARGN})
    endif()
  endforeach()
endfunction()


#-----------------------------------------------------------------------------#
#
# Internal routine to figure out if a list contains
# CUDA source code.  Returns empty or the list of CUDA files in the var
#
function(cuda_rdc_sources_contains_cuda var)
  set(_result)
  foreach(_source ${ARGN})
    get_source_file_property(_iscudafile "${_source}" LANGUAGE)
    if(_iscudafile STREQUAL "CUDA")
      list(APPEND _result "${_source}")
    elseif(NOT _iscudafile)
      get_filename_component(_ext "${_source}" LAST_EXT)
      if(_ext STREQUAL ".cu")
        list(APPEND _result "${_source}")
      endif()
    endif()
  endforeach()
  set(${var} "${_result}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
#
# Internal routine to figure out if a target already contains
# CUDA source code.  Returns empty or list of CUDA files in the OUTPUT_VARIABLE
#
function(cuda_rdc_lib_contains_cuda OUTPUT_VARIABLE target)
  cuda_rdc_strip_alias(target ${target})

  get_target_property(_targettype ${target} CUDA_RDC_LIBRARY_TYPE)
  if(_targettype)
    # The target is one of the components of a library with CUDA separatable code,
    # no need to check the source files.
    set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
  else()
    get_target_property(_target_sources ${target} SOURCES)
    cuda_rdc_sources_contains_cuda(_contains_cuda ${_target_sources})
    set(${OUTPUT_VARIABLE} ${_contains_cuda} PARENT_SCOPE)
  endif()
endfunction()

#-----------------------------------------------------------------------------#
#
# Generate an empty .cu file to transform the library to a CUDA library
#
function(cuda_rdc_generate_empty_cu_file emptyfilenamevar target)
  set(_stub "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${target}_emptyfile.cu")
  if(NOT EXISTS ${_stub})
    file(WRITE "${_stub}" "/* intentionally empty. */")
  endif()
  set(${emptyfilenamevar} ${_stub} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
#
# Transfer the setting \${what} (both the PUBLIC and INTERFACE version) to from library \${fromlib} to the library \${tolib} that depends on it
#
function(cuda_rdc_transfer_setting fromlib tolib what)
  get_target_property(_temp ${fromlib} ${what})
  if(_temp)
    cmake_language(CALL target_${what} ${tolib} PUBLIC ${_temp})
  endif()
  get_target_property(_temp ${fromlib} INTERFACE_${what})
  if(_temp)
    cmake_language(CALL target_${what} ${tolib} PUBLIC ${_temp})
  endif()
endfunction()

#-----------------------------------------------------------------------------#
# cuda_rdc_add_library
#
# Add a library taking into account whether it contains
# or depends on separatable CUDA code.  If it contains
# cuda code, it will be marked as "separatable compilation"
# (i.e. request "Relocatable device code")
#
function(cuda_rdc_add_library target)

  cuda_rdc_get_sources_and_options(_sources _cmake_options _options ${ARGN})

  if (_cmake_options STREQUAL "ALIAS")
    if(NOT ${ARGC} EQUAL 3)
      # Not sure what this syntax is ... pass the buck
      add_library(${target} ${ARGN})
      return()
    endif()
    # The aliased name must not be an alias, so no need to strip set
    set(alias_target ${_sources})
    get_target_property(_targettype ${alias_target} CUDA_RDC_LIBRARY_TYPE)

    string(REGEX MATCH ".*::" _scopename ${target})

    if(NOT _targettype OR NOT _scopename)
      # Not sure what this syntax is ... pass the buck
      add_library(${target} ${ARGN})
      return()
    endif()

    string(REGEX REPLACE "::$" "" _scopename ${_scopename})

    # No alias setup for the object library which are not exported.
    set(_libs ${alias_target})
    set(_commom_props "")
    foreach(_type "FINAL" "MIDDLE" "STATIC")
      get_target_property(_lib ${alias_target} "CUDA_RDC_${_type}_LIBRARY")
      if(_lib)
        cuda_rdc_strip_alias(_lib ${_lib})
        set(scopedname "${_scopename}::${_lib}")
        if(NOT TARGET ${scopedname})
          add_library(${scopedname} ALIAS ${_lib})
        endif()
        list(APPEND _libs ${_lib})
        list(APPEND _commom_props "CUDA_RDC_${_type}_LIBRARY" ${scopedname})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _libs)
    foreach(_lib ${_libs})
      set_target_properties(${_lib} PROPERTIES ${_commom_props})
    endforeach()
    return()
  endif()

  cuda_rdc_sources_contains_cuda(_cuda_sources ${_sources})

  # Whether we need the special code or not is actually dependent on information
  # we don't have ... yet
  # - whether the user request CUDA_SEPARABLE_COMPILATION
  # - whether the library depends on a library with CUDA_SEPARABLE_COMPILATION code.
  # I.e. this should really be done at generation time.
  # So in the meantime we use rely on the user to call this routine
  # only in the case where they want the CUDA device code to be compiled
  # as "relocatable device code"

  if(NOT CMAKE_CUDA_COMPILER OR NOT _cuda_sources)
    add_library(${target} ${ARGN})
    return()
  endif()

  cmake_parse_arguments(_ADDLIB_PARSE
    "STATIC;SHARED;MODULE;OBJECT"
    ""
    ""
    ${ARGN}
  )
  set(_lib_requested_type "SHARED")
  set(_cudaruntime_requested_type "Shared")
  set(_staticsuf "_static")
  if((NOT BUILD_SHARED_LIBS AND NOT _ADDLIB_PARSE_SHARED AND NOT _ADDLIB_PARSE_MODULE)
      OR _ADDLIB_PARSE_STATIC)
    set(_lib_requested_type "STATIC")
    set(_cudaruntime_requested_type "Static")
    set(_staticsuf "")
  endif()
  if(_ADDLIB_PARSE_MODULE)
    add_library(${target} ${ARGN})
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RUNTIME_LIBRARY ${_cudaruntime_requested_type}
    )
    return()
  endif()
  if(_ADDLIB_PARSE_OBJECT)
    message(FATAL_ERROR "cuda_rdc_add_library does not support OBJECT library")
  endif()

  ## OBJECTS ##

  add_library(${target}_objects OBJECT ${_ADDLIB_PARSE_UNPARSED_ARGUMENTS})
  set(_object_props
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY ${_cudaruntime_requested_type}
  )
  if(_lib_requested_type STREQUAL "SHARED")
    list(APPEND _object_props
      POSITION_INDEPENDENT_CODE ON
    )
  endif()
  set_target_properties(${target}_objects PROPERTIES ${_object_props})

  ## MIDDLE (main library) ##

  add_library(${target} ${_lib_requested_type}
    $<TARGET_OBJECTS:${target}_objects>
  )
  set(_common_props
    ${_object_props}
    LINKER_LANGUAGE CUDA
    CUDA_RDC_FINAL_LIBRARY ${target}_final
    CUDA_RDC_MIDDLE_LIBRARY ${target}
    CUDA_RDC_STATIC_LIBRARY ${target}${_staticsuf}
    CUDA_RDC_OBJECT_LIBRARY ${target}_objects
  )
  set_target_properties(${target} PROPERTIES
    ${_common_props}
    CUDA_RDC_LIBRARY_TYPE Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF # We really don't want nvlink called.
    EXPORT_PROPERTIES "CUDA_RDC_LIBRARY_TYPE;CUDA_RDC_FINAL_LIBRARY;CUDA_RDC_MIDDLE_LIBRARY;CUDA_RDC_STATIC_LIBRARY"
  )

  ## STATIC ##

  if(_staticsuf)
    add_library(${target}${_staticsuf} STATIC
      $<TARGET_OBJECTS:${target}_objects>
    )
    set_target_properties(${target}${_staticsuf} PROPERTIES
      ${_common_props}
      CUDA_RDC_LIBRARY_TYPE Static
      EXPORT_PROPERTIES "CUDA_RDC_LIBRARY_TYPE;CUDA_RDC_FINAL_LIBRARY;CUDA_RDC_MIDDLE_LIBRARY;CUDA_RDC_STATIC_LIBRARY"
    )
  endif()

  ## FINAL (dlink) ##

  # We need to use a dummy file as a library (per cmake) needs to contains
  # at least one source file.  The real content of the library will be
  # the cmake_device_link.o resulting from the execution of `nvcc -dlink`
  # Also non-cuda related test, for example `gtest_detail_Macros`,
  # will need to be linked again libcuda_rdc_final while a library
  # that the depends on and that uses Celeritas::Core (for example
  # libCeleritasTest.so) will need to be linked against `libceleritas`.
  # If both the middle and `_final` contains the `.o` files we would
  # then have duplicated symbols .  If both the middle and `_final`
  # library contained the result of `nvcc -dlink` then we would get
  # conflicting but duplicated *weak* symbols and here the symptoms
  # will be a crash during the cuda library initialization or a failure to
  # launch some kernels rather than a link error.
  cuda_rdc_generate_empty_cu_file(_emptyfilename ${target})
  add_library(${target}_final ${_lib_requested_type} ${_emptyfilename})
  set_target_properties(${target}_final PROPERTIES
    ${_common_props}
    LINK_DEPENDS $<TARGET_FILE:${target}${_staticsuf}>
    CUDA_RDC_LIBRARY_TYPE Final
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    EXPORT_PROPERTIES "CUDA_RDC_LIBRARY_TYPE;CUDA_RDC_FINAL_LIBRARY;CUDA_RDC_MIDDLE_LIBRARY;CUDA_RDC_STATIC_LIBRARY"
  )
  target_link_libraries(${target}_final PUBLIC ${target} PRIVATE CUDA::toolkit)
  target_link_options(${target}_final
    PRIVATE $<DEVICE_LINK:$<TARGET_FILE:${target}${_staticsuf}>>
  )
  add_dependencies(${target}_final ${target}${_staticsuf})
endfunction()

# Replacement for target_include_directories that is aware of
# the 4 libraries (objects, static, middle, final) libraries needed
# for a separatable CUDA library
function(cuda_rdc_target_include_directories target)
  if(NOT CMAKE_CUDA_COMPILER)
    target_include_directories(${ARGV})
    return()
  endif()

  cuda_rdc_strip_alias(target ${target})
  cuda_rdc_lib_contains_cuda(_contains_cuda ${target})

  if(_contains_cuda)
    get_target_property(_targettype ${target} CUDA_RDC_LIBRARY_TYPE)
    if(_targettype)
      get_target_property(_target_middle ${target} CUDA_RDC_MIDDLE_LIBRARY)
      get_target_property(_target_object ${target} CUDA_RDC_OBJECT_LIBRARY)
    endif()
  endif()
  if(_target_object)
    target_include_directories(${_target_object} ${ARGN})
  endif()
  if(_target_middle)
    cuda_rdc_strip_alias(_target_middle ${_target_middle})
    target_include_directories(${_target_middle} ${ARGN})
  else()
    target_include_directories(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#
# Replacement for target_compile_options that is aware of
# the 4 libraries (objects, static, middle, final) libraries needed
# for a separatable CUDA library
function(cuda_rdc_target_compile_options target)
  if(NOT CELERITAS_USE_CUDA)
    target_compile_options(${ARGV})
    return()
  endif()

  cuda_rdc_strip_alias(target ${target})
  cuda_rdc_lib_contains_cuda(_contains_cuda ${target})

  if(_contains_cuda)
    get_target_property(_targettype ${target} CUDA_RDC_LIBRARY_TYPE)
    if(_targettype)
      get_target_property(_target_middle ${target} CUDA_RDC_MIDDLE_LIBRARY)
      get_target_property(_target_object ${target} CUDA_RDC_OBJECT_LIBRARY)
    endif()
  endif()
  if(_target_object)
    target_compile_options(${_target_object} ${ARGN})
  endif()
  if(_target_middle)
    cuda_rdc_strip_alias(_target_middle ${_target_middle})
    target_compile_options(${_target_middle} ${ARGN})
  else()
    target_compile_options(${ARGV})
  endif()
endfunction()

#-----------------------------------------------------------------------------#
#
# Replacement for the install function that is aware of the 3 libraries
# (static, middle, final) libraries needed for a separatable CUDA library
#
function(cuda_rdc_install subcommand firstarg)
  if(NOT subcommand STREQUAL "TARGETS" OR NOT TARGET ${firstarg})
    install(${ARGV})
    return()
  endif()
  set(_targets ${firstarg})
  list(POP_FRONT ARGN _next)
  while(TARGET ${_next})
    list(APPEND _targets ${_next})
    list(POP_FRONT ${ARGN} _next)
  endwhile()
  # At this point all targets are in ${_targets} and ${_next} is the first non target and ${ARGN} is the rest.
  foreach(_target_elem ${_targets})
    cuda_rdc_strip_alias(_prop_target ${_target_elem})
    get_target_property(_lib_target_type ${_prop_target} TYPE)
    if(NOT _lib_target_type STREQUAL "INTERFACE_LIBRARY")
      get_target_property(_targettype ${_prop_target} CUDA_RDC_LIBRARY_TYPE)
      if(_targettype)
        get_target_property(_target_final ${_prop_target} CUDA_RDC_FINAL_LIBRARY)
        cuda_rdc_strip_alias(_target_final ${_target_final})
        get_target_property(_target_middle ${_prop_target} CUDA_RDC_MIDDLE_LIBRARY)
        cuda_rdc_strip_alias(_target_middle ${_target_middle})
        get_target_property(_target_static ${_prop_target} CUDA_RDC_STATIC_LIBRARY)
        cuda_rdc_strip_alias(_target_static ${_target_static})
        list(APPEND _toinstall ${_target_final})
        if(NOT _target_static STREQUAL _prop_target)
          list(APPEND _toinstall ${_target_static})
        endif()
        if(NOT _target_middle STREQUAL _prop_target AND NOT _target_middle STREQUAL _target_static)
          list(APPEND _toinstall ${_target_middle})
        endif()
      endif()
    endif()
    list(APPEND _toinstall ${_target_elem})
  endforeach()
  install(TARGETS ${_toinstall} ${_next} ${ARGN})
  endfunction()

#-----------------------------------------------------------------------------#
# Return TRUE if 'lib' depends/uses directly or indirectly the library `potentialdepend`
function(cuda_rdc_depends_on OUTVARNAME lib potentialdepend)
  set(${OUTVARNAME} FALSE PARENT_SCOPE)
  if(TARGET ${lib} AND TARGET ${potentialdepend})
    set(lib_link_libraries "")
    get_target_property(_lib_target_type ${lib} TYPE)
    if(NOT _lib_target_type STREQUAL "INTERFACE_LIBRARY")
      get_target_property(lib_link_libraries ${lib} LINK_LIBRARIES)
    endif()
    if(NOT lib_link_libraries)
      return()
    endif()
    foreach(linklib ${lib_link_libraries})
      if(linklib STREQUAL potentialdepend)
        set(${OUTVARNAME} TRUE PARENT_SCOPE)
        return()
      endif()
      cuda_rdc_depends_on(${OUTVARNAME} ${linklib} ${potentialdepend})
      if(${OUTVARNAME})
        set(${OUTVARNAME} ${${OUTVARNAME}} PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()
endfunction()

#-----------------------------------------------------------------------------#
# Return the 'real' target name whether the output is an alias or not.
function(cuda_rdc_strip_alias OUTVAR target)
  if(TARGET ${target})
    get_target_property(_target_alias ${target} ALIASED_TARGET)
  endif()
  if(TARGET ${_target_alias})
    set(target ${_target_alias})
  endif()
  set(${OUTVAR} ${target} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
# Return the middle/shared library of the target, if any.
macro(cuda_rdc_get_library_middle_target outvar target)
  get_target_property(_target_type ${target} TYPE)
  if(NOT _target_type STREQUAL "INTERFACE_LIBRARY")
    get_target_property(${outvar} ${target} CUDA_RDC_MIDDLE_LIBRARY)
  else()
    set(${outvar} ${target})
  endif()
endmacro()

#-----------------------------------------------------------------------------#
# Retrieve the "middle" library, i.e. given a target, the
# target name to be used as input to the linker of dependent libraries.
function(cuda_rdc_use_middle_lib_in_property target property)
  get_target_property(_target_libs ${target} ${property})

  set(_new_values)
  foreach(_lib ${_target_libs})
    set(_newlib ${_lib})
    # Simplistic treatement of `$<LINK_ONLY:...>`
    string(REGEX REPLACE "\\\$<LINK_ONLY:(.*)>" "\\1" _stripped_lib ${_lib})
    if(_stripped_lib STREQUAL _lib)
      set(_stripped_lib)
    else()
      set(_lib ${_stripped_lib})
    endif()
    if(TARGET ${_lib})
      cuda_rdc_strip_alias(_lib ${_lib})
      cuda_rdc_get_library_middle_target(_libmid ${_lib})
      if(_libmid)
        if(_stripped_lib)
          set(_newlib "$<LINK_ONLY:${_libmid}>")
        else()
          set(_newlib ${_libmid})
        endif()
      endif()
    endif()
    list(APPEND _new_values ${_newlib})
  endforeach()

  if(_new_values)
    set_target_properties(${target} PROPERTIES
      ${property} "${_new_values}"
    )
  endif()
endfunction()

#-----------------------------------------------------------------------------#
# Return the most derived "separatable cuda" library the target depends on.
# If two or more cuda library are independent, we return both and the calling executable
# should be linked with nvcc -dlink.
function(cuda_rdc_find_final_library OUTLIST flat_dependency_list)
  set(_result "")
  foreach(_lib ${flat_dependency_list})
    if(NOT _result)
      list(APPEND _result ${_lib})
    else()
      set(_newresult "")
      foreach(_reslib ${_result})
        cuda_rdc_depends_on(_depends_on ${_reslib} ${_lib})
        if(${_depends_on})
          # The library in the result depends/uses the library we are looking at,
          # let's keep the ones from result
          set(_newresult ${_result})
          break()
          # list(APPEND _newresult ${_reslib})
        else()
          cuda_rdc_depends_on(_depends_on ${_lib} ${_reslib})
          if(${_depends_on})
            # We are in the opposite case, let's keep the library we are looking at
            list(APPEND _newresult ${_lib})
          else()
            # Unrelated keep both
            list(APPEND _newresult ${_reslib})
            list(APPEND _newresult ${_lib})
          endif()
        endif()
      endforeach()
      set(_result ${_newresult})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _result)
  set(_final_result "")
  foreach(_lib ${_result})
    if(TARGET ${_lib})
      get_target_property(_lib_target_type ${_lib} TYPE)
      if(NOT _lib_target_type STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_final_lib ${_lib} CUDA_RDC_FINAL_LIBRARY)
        if(_final_lib)
          list(APPEND _final_result ${_final_lib})
        endif()
      endif()
    endif()
  endforeach()
  set(${OUTLIST} ${_final_result} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
#
#  Check which CUDA runtime is needed for a given (dependent) library.
function(cuda_rdc_check_cuda_runtime OUTVAR library)

  get_target_property(_runtime_setting ${library} CUDA_RUNTIME_LIBRARY)
  if(NOT _runtime_setting)
    # We could get more exact information by using:
    #  file(GET_RUNTIME_DEPENDENCIES LIBRARIES ${_lib_loc} UNRESOLVED_DEPENDENCIES_VAR _lib_dependcies)
    # but we get
    #   You have used file(GET_RUNTIME_DEPENDENCIES) in project mode.  This is
    #     probably not what you intended to do.
    # On the other hand, if the library is using (relocatable) CUDA code and
    # the shared run-time library and we don't have the scafolding libraries
    # (shared/static/final) then this won't work well. i.e. if we were to detect this
    # case we probably need to 'error out'.
    get_target_property(_cuda_library_type ${library} CUDA_RDC_LIBRARY_TYPE)
    get_target_property(_cuda_find_library ${library} CUDA_RDC_FINAL_LIBRARY)
    if(_cuda_library_type STREQUAL "Shared")
      set_target_properties(${library} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
      set(_runtime_setting "Shared")
    elseif(NOT _cuda_find_library)
      set_target_properties(${library} PROPERTIES CUDA_RUNTIME_LIBRARY "None")
      set(_runtime_setting "None")
    else()
      # If we have a final library then the library is shared.
      set_target_properties(${library} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
      set(_runtime_setting "Shared")
    endif()
  endif()

  set(${OUTVAR} ${_runtime_setting} PARENT_SCOPE)
endfunction()


#-----------------------------------------------------------------------------#
# Replacement for target_link_libraries that is aware of
# the 3 libraries (static, middle, final) libraries needed
# for a separatable CUDA library
function(cuda_rdc_target_link_libraries target)
  if(NOT CMAKE_CUDA_COMPILER)
    target_link_libraries(${ARGV})
    return()
  endif()

  cuda_rdc_strip_alias(target ${target})

  # Reset the cached dependency list
  set_property(TARGET ${target} PROPERTY CUDA_RDC_CACHED_LIB_DEPENDENCIES)

  cuda_rdc_lib_contains_cuda(_contains_cuda ${target})

  set(_target_final ${target})
  set(_target_middle ${target})
  if(_contains_cuda)
    get_target_property(_targettype ${target} CUDA_RDC_LIBRARY_TYPE)
    if(_targettype)
      get_target_property(_target_final ${target} CUDA_RDC_FINAL_LIBRARY)
      get_target_property(_target_middle ${target} CUDA_RDC_MIDDLE_LIBRARY)
      get_target_property(_target_object ${target} CUDA_RDC_OBJECT_LIBRARY)
    endif()
  endif()

  # Set now to let target_link_libraries do the argument parsing
  cuda_rdc_strip_alias(_target_middle ${_target_middle})
  target_link_libraries(${_target_middle} ${ARGN})

  cuda_rdc_use_middle_lib_in_property(${_target_middle} INTERFACE_LINK_LIBRARIES)
  cuda_rdc_use_middle_lib_in_property(${_target_middle} LINK_LIBRARIES)

  if(_target_object)
    target_link_libraries(${_target_object} ${ARGN})
    cuda_rdc_use_middle_lib_in_property(${_target_object} INTERFACE_LINK_LIBRARIES)
    cuda_rdc_use_middle_lib_in_property(${_target_object} LINK_LIBRARIES)
  endif()

  cuda_rdc_cuda_gather_dependencies(_alldependencies ${target})
  cuda_rdc_find_final_library(_finallibs "${_alldependencies}")

  get_target_property(_target_type ${target} TYPE)
  if(_target_type STREQUAL "EXECUTABLE"
     OR _target_type STREQUAL "MODULE_LIBRARY")
    list(LENGTH _finallibs _final_count)
    if(_contains_cuda)
      if(${_final_count} GREATER 0)
        # If there is at least one final library this means that we
        # have somewhere some "separable" nvcc compilations
        set_target_properties(${target} PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
        )
      endif()
    elseif(${_final_count} EQUAL 1)
      set_target_properties(${target} PROPERTIES
        # If cmake detects that a library depends/uses a static library
        # linked with CUDA, it will turn CUDA_RESOLVE_DEVICE_SYMBOLS ON
        # leading to a call to nvlink.  If we let this through (at least
        # in case of Celeritas) we would need to add the DEVICE_LINK options
        # also on non cuda libraries (that we detect depends on a cuda library).
        # Note: we might be able to move this to cuda_rdc_target_link_libraries
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      )
      get_target_property(_final_target_type ${_finallibs} TYPE)

      get_target_property(_final_runtime ${_finallibs} CUDA_RUNTIME_LIBRARY)
      if(_final_runtime STREQUAL "Shared")
        set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
      endif()

      if(_final_target_type STREQUAL "STATIC_LIBRARY")
        # for static libraries we need to list the libraries a second time (to resolve symbol from the final library)
        get_target_property(_current_link_libraries ${target} LINK_LIBRARIES)
        set_property(TARGET ${target} PROPERTY LINK_LIBRARIES ${_current_link_libraries} ${_finallibs} ${_current_link_libraries} )
      else()
        # We could have used:
        #    target_link_libraries(${target} PUBLIC ${_finallibs})
        # but target_link_libraries must used either all plain arguments or all plain
        # keywords and at the moment I don't know how to detect which of the 2 style the
        # user used.
        # Maybe we could use:
        #     if(ARGV1 MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
        # or simply keep the following:
        get_target_property(_current_link_libraries ${target} LINK_LIBRARIES)
        set_property(TARGET ${target} PROPERTY LINK_LIBRARIES ${_current_link_libraries} "$<LINK_LIBRARY:rdc_no_as_needed,${_finallibs}>" )
      endif()
    elseif(${_final_count} GREATER 1)
      # turn into CUDA executable.
      set(_contains_cuda TRUE)
      cuda_rdc_generate_empty_cu_file(_emptyfilename ${target})
      target_sources(${target} PRIVATE ${_emptyfilename})
      # If there is at least one final library this means that we
      # have somewhere some "separable" nvcc compilations
      set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
    endif()
    # nothing to do if there is no final library (i.e. no use of CUDA at all?)
  endif()

  if(_contains_cuda)
    set(_need_to_use_shared_runtime FALSE)
    get_target_property(_current_runtime_setting ${target} CUDA_RUNTIME_LIBRARY)
    if(_current_runtime_setting)
       set(_target_runtime_setting ${_current_runtime_setting})
    endif()
    cuda_rdc_cuda_gather_dependencies(_flat_target_link_libraries ${_target_middle})
    cuda_rdc_strip_alias(_target_final ${_target_final})
    foreach(_lib ${_flat_target_link_libraries})
      get_target_property(_lib_target_type ${_lib} TYPE)
      if(NOT _lib_target_type STREQUAL "INTERFACE_LIBRARY")
        cuda_rdc_check_cuda_runtime(_lib_runtime_setting ${_lib})
        if(NOT _need_to_use_shared_runtime AND _lib_runtime_setting STREQUAL "Shared")
          set(_need_to_use_shared_runtime TRUE)
        endif()
        if(NOT _target_runtime_setting)
          if(_lib_runtime_setting)
            set(_target_runtime_setting ${_lib_runtime_setting})
          endif()
        else()
          if(_lib_runtime_setting AND NOT (_target_runtime_setting STREQUAL _lib_runtime_setting))
            # We need to match the dependent library since we can not change it.
            set(_target_runtime_setting ${_lib_runtime_setting})
          endif()
        endif()
        if(NOT _current_runtime_setting)
          set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY ${_target_runtime_setting})
        endif()
        get_target_property(_libstatic ${_lib} CUDA_RDC_STATIC_LIBRARY)
        if(_target_type STREQUAL "EXECUTABLE")
           # We need to explicit list the RDC library without the compiler might complain with:
           #    error adding symbols: DSO missing from command line
           # This DSO missing from command line message will be displayed when the linker
           # does not find the required symbol with it’s normal search but the symbol is
           # available in one of the dependencies of a directly specified dynamic library.
           # In the past the linker considered symbols in dependencies of specified languages
           # to be available. But that changed in some later version and now the linker
           # enforces a more strict view of what is available.
           target_link_libraries(${_target_final} ${_lib})
        endif()
        if(TARGET ${_libstatic})
          target_link_options(${_target_final}
            PRIVATE
            $<DEVICE_LINK:$<TARGET_FILE:${_libstatic}>>
          )
          set_property(TARGET ${_target_final} APPEND
            PROPERTY LINK_DEPENDS $<TARGET_FILE:${_libstatic}>
          )

          # Also pass on the the options and definitions.
          cuda_rdc_transfer_setting(${_libstatic} ${_target_final} COMPILE_OPTIONS)
          cuda_rdc_transfer_setting(${_libstatic} ${_target_final} COMPILE_DEFINITIONS)
          cuda_rdc_transfer_setting(${_libstatic} ${_target_final} LINK_OPTIONS)

          add_dependencies(${_target_final} ${_libstatic})
        endif()
      endif()
    endforeach()


    if(_need_to_use_shared_runtime)
      get_target_property(_current_runtime ${target} CUDA_RUNTIME_LIBRARY)
      if(NOT _current_runtime STREQUAL "Shared")
        set_target_properties(${_target_middle} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
        set_target_properties(${_target_final} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
        set_target_properties(${_target_object} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
      endif()
    endif()
  else() # We could restrict to the case where the dependent is a static library ... maybe
    set_target_properties(${target} PROPERTIES
      # If cmake detects that a library depends/uses a static library
      # linked with CUDA, it will turn CUDA_RESOLVE_DEVICE_SYMBOLS ON
      # leading to a call to nvlink.  If we let this through (at least
      # in case of Celeritas) we would need to add the DEVICE_LINK options
      # also on non cuda libraries (that we detect depends on a cuda library).
      # Note: we might be able to move this to cuda_rdc_target_link_libraries
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    if(NOT _target_type STREQUAL "EXECUTABLE")
      get_target_property(_current_runtime ${target} CUDA_RUNTIME_LIBRARY)
      if(NOT _current_runtime STREQUAL "Shared")
        set(_need_to_use_shared_runtime FALSE)
        foreach(_lib ${_alldependencies})
          cuda_rdc_check_cuda_runtime(_runtime ${_lib})
          if(_runtime STREQUAL "Shared")
            set(_need_to_use_shared_runtime TRUE)
            break()
          endif()
        endforeach()
        # We do not yet treat the case where the dependent library is Static
        # and the current one is Shared.
        if(${_need_to_use_shared_runtime})
          set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY "Shared")
        endif()
      endif()
    endif()
  endif()

endfunction()

#-----------------------------------------------------------------------------#
#
# Return a flat list of all the direct and indirect dependencies of 'target'
# which are libraries containing CUDA separatable code.
#
function(cuda_rdc_cuda_gather_dependencies outlist target)
  if(NOT TARGET ${target})
    return()
  endif()
  get_target_property(_cached_dependencies ${target} CUDA_RDC_CACHED_LIB_DEPENDENCIES)
  if (_cached_dependencies)
     set(${outlist} ${_cached_dependencies} PARENT_SCOPE)
     return()
  endif()
  cuda_rdc_strip_alias(target ${target})
  get_target_property(_target_type ${target} TYPE)
  set(_deplist)
  if(NOT _target_type STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_target_link_libraries ${target} LINK_LIBRARIES)
    if(_target_link_libraries)
      foreach(_lib ${_target_link_libraries})
        cuda_rdc_strip_alias(_lib ${_lib})
        set(_libmid)
        if(TARGET ${_lib})
          cuda_rdc_get_library_middle_target(_libmid ${_lib})
        endif()
        if(TARGET ${_libmid})
          list(APPEND _deplist ${_libmid})
        endif()
        # and recurse
        set(_midlist)
        set(_before ${_midlist})
        cuda_rdc_cuda_gather_dependencies(_midlist ${_lib})
        list(APPEND _deplist ${_midlist})
      endforeach()
    endif()
  endif()
  list(REMOVE_DUPLICATES _deplist)
  list(APPEND ${outlist} ${_deplist})
  list(REMOVE_DUPLICATES ${outlist})
  set_target_properties(${target} PROPERTIES CUDA_RDC_CACHED_LIB_DEPENDENCIES "${_deplist}")
  set(${outlist} ${${outlist}} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
