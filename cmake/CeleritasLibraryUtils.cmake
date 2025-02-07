#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasLibraryUtils
---------------------

CMake library functions for Celeritas. This *must* be included after
configurations are set up. Most of these are wrappers that forward directly to
CMake or alternatively to CudaRdc.

.. command:: celeritas_add_src_library

  Add a library that correctly links against CUDA relocatable device code, has
  the ``Celeritas::`` aliases, is generated into the ``lib/`` build
  directory, and is installed with the project.

.. command:: celeritas_add_test_library

  Add a test-only library that uses RDC but does not get installed.

.. command:: celeritas_get_cuda_source_args

  Get a list of all source files in the argument that are CUDA language.

    celeritas_get_cuda_source_args(<var> [<source> ...])

.. command:: celeritas_add_library

  Add a library that correctly links against CUDA relocatable device code.

.. command:: celeritas_install

  Install library that correctly deal with CUDA relocatable device code.

.. command:: celeritas_set_target_properties

  Install library that correctly deal with CUDA relocatable device code extra
  libraries.

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

  Specify libraries or flags to use when linking a given target and/or its
  dependents, taking in account the extra targets (see
  celeritas_rdc_add_library) needed to support CUDA relocatable device code.

    ::

      celeritas_target_link_libraries(<target>
        <PRIVATE|PUBLIC|INTERFACE> <item>...
        [<PRIVATE|PUBLIC|INTERFACE> <item>...]...))

  Usage requirements from linked library targets will be propagated to all four
  targets. Usage requirements of a target's dependencies affect compilation of
  its own sources. In the case that ``<target>`` does not contain CUDA code, the
  command decays to ``target_link_libraries``.

  See ``target_link_libraries`` for additional detail.


.. command:: celeritas_target_include_directories

  Add include directories to a target.

    ::

      celeritas_target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
        <INTERFACE|PUBLIC|PRIVATE> [items1...]
        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  Specifies include directories to use when compiling a given target. The named
  <target> must have been created by a command such as
  celeritas_rdc_add_library(), add_executable() or add_library(), and can be
  used with an ALIAS target. It is aware of the 4 underlying targets (objects,
  static, middle, final) present when the input target was created
  celeritas_rdc_add_library() and will propagate the include directories to all
  four. In the case that ``<target>`` does not contain CUDA code, the command
  decays to ``target_include_directories``.

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

.. command:: celeritas_polysource_append

  Add C++ and CUDA/HIP source files based on the enabled options.

    celeritas_polysource_append(SOURCES my/Class)

#]=======================================================================]
include_guard(GLOBAL)

if(NOT DEFINED CELERITAS_USE_VecGeom)
  message(FATAL_ERROR
    "This file can only be included after options are defined"
  )
endif()

# Include RDC utils if not included already
include(CudaRdcUtils)

#-----------------------------------------------------------------------------#
# Wrapper functions

if(NOT (CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA))
  # Forward all arguments direcly to CMake builtins
  macro(celeritas_add_library)
    add_library(${ARGV})
  endmacro()
  macro(celeritas_set_target_properties)
    set_target_properties(${ARGV})
  endmacro()
  macro(celeritas_install)
    install(${ARGV})
  endmacro()
  macro(celeritas_target_link_libraries)
    target_link_libraries(${ARGV})
  endmacro()
  macro(celeritas_target_include_directories)
    target_include_directories(${ARGV})
  endmacro()
  macro(celeritas_target_compile_options)
    target_compile_options(${ARGV})
  endmacro()
else()
  # Forward all arguments to RDC utility wrappers
  macro(celeritas_add_library)
    cuda_rdc_add_library(${ARGV})
  endmacro()
  macro(celeritas_set_target_properties)
    cuda_rdc_set_target_properties(${ARGV})
  endmacro()
  macro(celeritas_install)
    cuda_rdc_install(${ARGV})
  endmacro()
  macro(celeritas_target_link_libraries)
    cuda_rdc_target_link_libraries(${ARGV})
  endmacro()
  macro(celeritas_target_include_directories)
    cuda_rdc_target_include_directories(${ARGV})
  endmacro()
  macro(celeritas_target_compile_options)
    cuda_rdc_target_compile_options(${ARGV})
  endmacro()
endif()

#-----------------------------------------------------------------------------#

function(celeritas_get_cuda_source_args var ${ARGN})
  # NOTE: this is the only function that uses CudaRdcUtils if VecGeom+CUDA is
  # disabled: it's only needed when building with HIP
  cuda_rdc_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_rdc_sources_contains_cuda(_cuda_sources ${_sources})
  set(${var} ${_cuda_sources} PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_add_src_library target)
  if(CELERITAS_USE_HIP)
    celeritas_get_cuda_source_args(_cuda_sources ${ARGN})
    if(_cuda_sources)
      # When building Celeritas libraries, we put HIP/CUDA files in shared .cu
      # suffixed files. Override the language if using HIP.
      set_source_files_properties(
        ${_cuda_sources}
        PROPERTIES LANGUAGE HIP
      )
    endif()
  endif()

  celeritas_add_library(${target} ${ARGN})

  # Add Celeritas:: namespace alias
  celeritas_add_library(Celeritas::${target} ALIAS ${target})

  # Build all targets in lib/
  set(_props
    ARCHIVE_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}"
  )

  if(CELERITAS_USE_ROOT)
    # Require PIC for static libraries, including "object" RDC lib
    list(APPEND _props
      POSITION_INDEPENDENT_CODE ON
    )
  endif()

  celeritas_set_target_properties(${target} PROPERTIES ${_props})

  # Install all targets to lib/
  celeritas_install(TARGETS ${target}
    EXPORT celeritas-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_add_test_library target)
  if(CELERITAS_USE_HIP)
    celeritas_get_cuda_source_args(_cuda_sources ${ARGN})
    if(_cuda_sources)
      # When building Celeritas libraries, we put HIP/CUDA files in shared .cu
      # suffixed files. Override the language if using HIP.
      set_source_files_properties(
        ${_cuda_sources}
        PROPERTIES LANGUAGE HIP
      )
    endif()
  endif()

  celeritas_add_library(${target} ${ARGN})

  if(CELERITAS_USE_ROOT)
    # Require PIC for static libraries, including "object" RDC lib
    celeritas_set_target_properties(${target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
    )
  endif()
endfunction()

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

macro(celeritas_polysource_append var filename_we)
  list(APPEND ${var} "${filename_we}.cc")
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND ${var} "${filename_we}.cu")
  endif()
endmacro()

#-----------------------------------------------------------------------------#
