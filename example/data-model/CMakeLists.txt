#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18...4.1)
project(CeleritasMinimalExample VERSION 0.0.1 LANGUAGES CXX)

find_package(Celeritas 0.6 REQUIRED)

add_executable(minimal minimal.cc)
# Use special linking macros to support CUDA+VecGeom when used
celeritas_target_link_libraries(minimal
  Celeritas::celeritas
)
