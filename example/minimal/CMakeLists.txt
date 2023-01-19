#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasMinimalExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

#-----------------------------------------------------------------------------#

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../_cmake")
find_package(Celeritas 0.2 REQUIRED)

include(CeleritasLibrary)

add_executable(minimal minimal.cc)
celeritas_target_link_libraries(minimal
  Celeritas::celeritas
)

#-----------------------------------------------------------------------------#
