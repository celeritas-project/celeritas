#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18...4.1)
project(CeleritasAccelExample VERSION 0.0.1 LANGUAGES CXX)

find_package(Celeritas 0.7 REQUIRED)
find_package(Geant4 REQUIRED)

if(NOT CELERITAS_USE_Geant4)
  message(SEND_ERROR "This example requires Geant4 support "
    "to be enabled in Celeritas")
endif()

add_executable(simple-offload simple-offload.cc)
celeritas_target_link_libraries(simple-offload
  Celeritas::G4
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.0)
  message(WARNING "Tracking manager offload requires Geant4 11.0 or higher")
endif()

add_executable(trackingmanager-offload trackingmanager-offload.cc)
celeritas_target_link_libraries(trackingmanager-offload
  Celeritas::G4
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.1)
  message(WARNING "Fast simulation offload requires Geant4 11.1 or higher")
endif()

add_executable(fastsim-offload fastsim-offload.cc)
celeritas_target_link_libraries(fastsim-offload
  Celeritas::G4
  ${Geant4_LIBRARIES}
)

# END EXAMPLE CODE

#-----------------------------------------------------------------------------#
# Add tests for CI
#-----------------------------------------------------------------------------#

include(CTest)
include(CeleritasUtils)
include(CeleritasG4Tests)

celeritas_get_g4env(_g4data_env)
celeritas_get_disable_device(CELER_DISABLE_DEVICE)

#-----------------------------------------------------------------------------#

# Some features require *at least* major version 11 or minor version 11.1
set(_needs_g4_11)
if(Geant4_VERSION VERSION_LESS 11.0)
  set(_needs_g4_11 DISABLE)
endif()
set(_needs_g4_11_1)
if(Geant4_VERSION VERSION_LESS 11.1)
  set(_needs_g4_11_1 DISABLE)
endif()

celeritas_g4_add_tests(simple-offload)
celeritas_g4_add_tests(trackingmanager-offload ${_needs_g4_11})
celeritas_g4_add_tests(fastsim-offload ${_needs_g4_11_1})
