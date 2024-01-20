#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

Run with `cmake -P 2>&1`, optionally passing -DKEY=<val>

See:
https://cmake.org/cmake/help/latest/command/cmake_host_system_information.html
for possible values.

#]=======================================================================]

if(NOT KEY)
  set(KEY NUMBER_OF_LOGICAL_CORES)
endif()
cmake_host_system_information(RESULT result QUERY ${KEY})
message(${result})

#-----------------------------------------------------------------------------#
