#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Findrocthrust
-------------

Find AMD's ROCm port of the Thrust algorithm library.

#]=======================================================================]

# On ROCm rocThrust requires rocPRIM
find_package(rocprim QUIET REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
find_package(rocthrust QUIET REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocthrust CONFIG_MODE)

#-----------------------------------------------------------------------------#
