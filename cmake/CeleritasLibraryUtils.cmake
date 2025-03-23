#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasLibraryUtils
---------------------

CMake library functions for Celeritas internals. This *must* be included after
configurations and CudaRdcUtils are set up. Most of these are wrappers that
forward directly to CMake or alternatively to CudaRdc.

.. command:: celeritas_add_src_library

  Add a library that correctly links against CUDA relocatable device code, has
  the ``Celeritas::`` aliases, is generated into the ``lib/`` build
  directory, automatically uses HIP for ``.cu`` files, and is installed with the
  project.

.. command:: celeritas_add_test_library

  Add a test-only library that uses RDC but does not get installed.

.. command:: celeritas_add_interface_library

  Add an interface library that correctly links against CUDA relocatable device
  code, has the ``Celeritas::`` aliases, and is installed with the project.
  Interface libraries should have capitalized names to distinguish them.

.. command:: celeritas_add_object_library

  Add an OBJECT library to reduce dependencies (e.g. includes) from other
  libraries. The source files MUST NOT be CUDA/HIP.

.. command:: celeritas_add_executable

  Create an executable and install it::

    celeritas_add_executable(<target> [<source> ...])

  The ``<target>`` is a unique identifier for the executable target. The actual
  executable name may end up with an .exe suffix (e.g. if it's windows). The
  executable will be built into the top-level ``bin`` directory, so all
  executables will sit side-by-side before installing.

  The ``<source>`` arguments are passed to CMake's builtin ``add_executable``
  command.

.. command:: celeritas_configure_file

  Configure to the build "include" directory for later installation::

    celeritas_configure_file(<input> <output> [ARGS...])

  The ``<input>`` must be a relative path to the current source directory, and
  the ``<output>` path is configured to the project build "include" directory.

.. command:: celeritas_polysource_append

  Add C++ and CUDA/HIP source files based on the enabled options.

    celeritas_polysource_append(SOURCES my/Class)

Helper functions
^^^^^^^^^^^^^^^^

.. command:: celeritas_get_cuda_source_args

  Get a list of all source files in the argument that are CUDA language.

    celeritas_get_cuda_source_args(<var> [<source> ...])

#]=======================================================================]
include_guard(GLOBAL)

if(NOT COMMAND celeritas_add_library)
  message(FATAL_ERROR
    "This file can only be included after CeleritasLibrary is loaded."
  )
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

  celeritas_add_library(${ARGV})

  if(CELERITAS_USE_ROOT)
    # Require PIC for static libraries, including "object" RDC lib
    celeritas_set_target_properties(${target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
    )
  endif()
endfunction()

#-----------------------------------------------------------------------------#

function(celeritas_add_interface_library target)
  # Interface libraries don't need RDC wrappers
  add_library(${target} INTERFACE ${ARGN})
  add_library(Celeritas::${target} ALIAS ${target})
  install(TARGETS ${target}
    EXPORT celeritas-targets
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

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
  add_executable(${ARGV})
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
