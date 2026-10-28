#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindThrust
----------

Find the Thrust algorithm library for CUDA. Note that HIP's installation may be
available under the name "rocthrust" but we can't handle that.

#]=======================================================================]

# CUDA stores things in lib64 which isn't in cmake's default search
set(_hints)
foreach(_dir IN ITEMS
  "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}"
  "${CUDAToolkit_ROOT}"
  "$ENV{CUDA_HOME}"
)
  if(_dir)
    list(APPEND _hints "${_dir}/lib64/cmake/thrust")
  endif()
endforeach()
unset(_dir)
list(REMOVE_DUPLICATES _hints)

find_package(Thrust QUIET CONFIG HINTS ${_hints})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust CONFIG_MODE)
unset(_hints)

#-----------------------------------------------------------------------------#
