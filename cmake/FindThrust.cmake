#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindThrust
----------

Find the Thrust algorithm library for CUDA. Note that HIP's installation may be
available under the name "rocthrust" but we can't handle that.

#]=======================================================================]

set(CMAKE_MESSAGE_LOG_LEVEL TRACE)

message("Before find_package: Thrust_DIR=${Thrust_DIR}")
find_package(Thrust CONFIG NAMES thrust CONFIGS thrust-config.cmake)
message("after config find: Thrust_DIR=${Thrust_DIR}")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust CONFIG_MODE)
message("After find_package: Thrust_DIR=${Thrust_DIR}")

#-----------------------------------------------------------------------------#
