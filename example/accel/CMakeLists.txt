#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasAccelExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")
find_package(Celeritas 0.3 REQUIRED)

if(NOT CELERITAS_USE_Geant4 OR CELERITAS_CORE_GEO STREQUAL "ORANGE")
  message(SEND_ERROR "This example requires Geant4 support "
    "to be enabled in Celeritas")
endif()

find_package(Geant4 REQUIRED)

include(CeleritasLibrary)

add_executable(simple-offload simple-offload.cc)
celeritas_target_link_libraries(simple-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)

add_executable(fastsim-offload fastsim-offload.cc)
celeritas_target_link_libraries(fastsim-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)
