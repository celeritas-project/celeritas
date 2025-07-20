#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasLibrary
----------------

These functions dispatch to the CudaRdcUtils if necessary (i.e., if CUDA and
VecGeom are enabled) and directly to CMake if not. It should be loaded
automatically by Celeritas in the main CMakeLists.txt, and for downstream
projects automatically by the CeleritasConfig.cmake file.

.. command:: celeritas_add_library

  CudaRdcUtils version of ``add_library``.

.. command:: celeritas_set_target_properties

  CudaRdcUtils version of ``set_target_properties``.

.. command:: celeritas_install

  CudaRdcUtils version of ``install``.

.. command:: celeritas_target_link_libraries

  CudaRdcUtils version of ``target_link_libraries``.

.. command:: celeritas_target_include_directories

  CudaRdcUtils version of ``target_include_directories``.

.. command:: celeritas_target_compile_options

  CudaRdcUtils version of ``target_compile_options``.


#]=======================================================================]
include_guard(GLOBAL)

if(NOT DEFINED CELERITAS_USE_VecGeom)
  message(FATAL_ERROR
    "This file can only be included after Celeritas options are defined"
  )
endif()

if(NOT (CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA))
  # Forward all arguments directly to CMake builtins
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
  if(NOT COMMAND cuda_rdc_add_library)
    message(FATAL_ERROR "This file MUST be used with CudaRdcUtils")
  endif()

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
