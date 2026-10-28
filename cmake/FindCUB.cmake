#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindCUB
----------

Find the CUB algorithm library for CUDA. Note that HIP's installation may be
available under the name "rocCUB" but we can't handle that.

#]=======================================================================]

set(_hints)
foreach(_dir IN ITEMS
  "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}"
  "${CUDAToolkit_ROOT}"
  "$ENV{CUDA_HOME}"
)
  if(_dir)
    list(APPEND _hints "${_dir}/lib64/cmake/cub")
  endif()
endforeach()
unset(_dir)
list(REMOVE_DUPLICATES _hints)

find_package(CUB QUIET CONFIG HINTS ${_hints})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB CONFIG_MODE)
unset(_hints)

#-----------------------------------------------------------------------------#
