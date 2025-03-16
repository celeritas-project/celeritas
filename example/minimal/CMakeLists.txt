#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasMinimalExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")

find_package(Celeritas 0.5 REQUIRED)

add_executable(minimal minimal.cc)

# Use Celeritas CUDA RDC macros to correctly link libraries using VecGeom
# NOTE: until vecgeom/g4vg get updated, we must use the Celeritas load path
include("${Celeritas_DIR}/CudaRdcUtils.cmake")

cuda_rdc_target_link_libraries(minimal
  Celeritas::celeritas
)
