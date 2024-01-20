#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasMinimalExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")
find_package(Celeritas 0.4 REQUIRED)

add_executable(minimal minimal.cc)

# Use special linking macros to support CUDA RDC across multiple libraries when
# using VecGeom.
include(CeleritasLibrary)
celeritas_target_link_libraries(minimal
  Celeritas::celeritas
)
