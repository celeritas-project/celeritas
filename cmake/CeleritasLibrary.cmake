#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasLibrary
----------------

This file is to be replaced with the wrappers from CeleritasLibraryUtils and
included automatically in Celeritas and downstream projects.

#]=======================================================================]

include("${CMAKE_CURRENT_LIST_DIR}/CudaRdcUtils.cmake")

message(AUTHOR_WARNING "CeleritasLibrary has been replaced by CudaRdcUtils.
Please include(CudaRdcUtils) and address the following warnings...")

macro(celeritas_add_library)
  message(AUTHOR_WARNING "Replace with cuda_rdc_add_library")
  cuda_rdc_add_library(${ARGV})
endmacro()

macro(celeritas_add_executable)
  message(AUTHOR_WARNING "Replace with cuda_rdc_add_executable")
  cuda_rdc_add_executable(${ARGV})
endmacro()

macro(celeritas_target_link_libraries)
  message(AUTHOR_WARNING "Replace with cuda_rdc_target_link_libraries")
  cuda_rdc_target_link_libraries(${ARGV})
endmacro()

macro(celeritas_target_compile_options)
  message(AUTHOR_WARNING "Replace with cuda_rdc_target_compile_options")
  cuda_rdc_target_compile_options(${ARGV})
endmacro()

macro(celeritas_set_target_properties)
  message(AUTHOR_WARNING "Replace with cuda_rdc_set_target_properties")
  cuda_rdc_set_target_properties(${ARGV})
endmacro()
