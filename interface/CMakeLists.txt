#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

.. command:: celeritas_swig_python_module

  ::

    celeritas_swig_python_module(<target> <module> [<source> ...])

  ``target``
    Name of the library module target to create.

  ``module``
    Path to the module interface file.

  ``source``
    Additional source files to build into the library.

#]=======================================================================]

include(UseSWIG)
set(CMAKE_SWIG_CXX_FLAGS "" CACHE STRING
  "C++ flags to pass to SWIG-generated code")

#-----------------------------------------------------------------------------#

function(celeritas_swig_python_module target module_source)
  # We're using C++
  set_property(SOURCE "${module_source}" PROPERTY CPLUSPLUS ON)
  # Propagate target include directories to the SWIG generate command
  set_property(SOURCE "${module_source}"
    PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)

  # Without 'builtin' 4.1 and 4.1.1 have unused variables
  set_property(SOURCE "${module_source}" APPEND_STRING
    PROPERTY COMPILE_OPTIONS "-builtin")

  # Create the library
  swig_add_library(${target}
    LANGUAGE Python
    TYPE MODULE
    SOURCES "${module_source}" ${ARGN}
  )

  # Add SWIG headers
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  celeritas_target_link_libraries(${target} PRIVATE Python::Module)
  if(CMAKE_SWIG_CXX_FLAGS)
    target_compile_options(${target} PRIVATE "${CMAKE_SWIG_CXX_FLAGS}")
  endif()

  # Install SWIG interface file
  install(FILES
    "${module_source}"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT development
  )

  # Install built module and wrapper files
  install(TARGETS ${target}
    DESTINATION interface
    COMPONENT runtime
  )
  get_property(_WRAPPERS TARGET ${target} PROPERTY SWIG_SUPPORT_FILES)
  install(FILES "${_WRAPPERS}"
    DESTINATION interface
    COMPONENT runtime
  )
endfunction()

#-----------------------------------------------------------------------------#

celeritas_swig_python_module(CeleritasPython celeritas.i)
add_library(Celeritas::Python ALIAS CeleritasPython)
celeritas_target_link_libraries(CeleritasPython PRIVATE Celeritas::celeritas)
