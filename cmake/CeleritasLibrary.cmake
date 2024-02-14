#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasLibrary
----------------

Deprecated file for use by downstream code. TODO: remove in v1.0.

#]=======================================================================]

include("${CMAKE_CURRENT_LIST_DIR}/CudaRdcUtils.cmake")

message(AUTHOR_WARNING "CeleritasLibrary has been replaced by CudaRdcUtils. Please include(CudaRdcUtils) and
replace celeritas_(add_library|add_executable|target_link_libraries) with
cuda_rdc_(...)")

macro(celeritas_add_library)
  message(AUTHOR_WARNING "Replace with cuda_rdc_add_library")
  cuda_rdc_add_library(${ARGV})
endmacro()

macro(celeritas_add_executable)
  message(AUTHOR_WARNING "Replace with cuda_rdc_add_executable")
  cuda_rdc_add_executable(${ARGV})
endmacro()

macro(celeritas_link_libraries)
  message(AUTHOR_WARNING "Replace with cuda_rdc_link_libraries")
  cuda_rdc_link_libraries(${ARGV})
endmacro()
