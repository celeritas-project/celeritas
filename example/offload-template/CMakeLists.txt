#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Minimal Geant4 application with Celeritas offloading capabilities
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(offload-template VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

# Only necessary for the CI
# Users can export Celeritas_ROOT or set CMAKE_INSTALL_PREFIX
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")

#-----------------------------------------------------------------------------#
# Find packages
find_package(Celeritas 0.5)
find_package(Geant4)

if(NOT Celeritas_FOUND)
  message(FATAL_ERROR
    "No Celeritas installation found.\n"
    "Export Celeritas_ROOT=/path/to/celeritas/install or use CMAKE_INSTALL_PREFIX"
  )
endif()

if(NOT CELERITAS_USE_Geant4)
  message(SEND_ERROR "This example requires Geant4 support "
    "to be enabled in Celeritas")
endif()

#-----------------------------------------------------------------------------#
# Add executable
add_executable(run-offload
  main.cc
  src/ActionInitialization.cc
  src/Celeritas.cc
  src/DetectorConstruction.cc
  src/EventAction.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SensitiveDetector.cc
)

target_include_directories(run-offload PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# Use Celeritas CUDA RDC macros to correctly link libraries using VecGeom
include(CudaRdcUtils)
cuda_rdc_target_link_libraries(run-offload
  ${Geant4_LIBRARIES}
  Celeritas::celeritas
  Celeritas::accel
)
