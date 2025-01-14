#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Minimal Geant4 application with Celeritas offloading capabilities
#-----------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.12...3.30)
project(offload-template)

#-----------------------------------------------------------------------------#
# Find packages
find_package(Celeritas 0.5)
find_package(Geant4 REQUIRED)

if(NOT Celeritas_FOUND)
  message(FATAL_ERROR
  "No Celeritas installation found.\n"
  "Export Celeritas_ROOT=/path/to/celeritas/install or use CMAKE_INSTALL_PREFIX")
endif()

#-----------------------------------------------------------------------------#
# Set up sources, include directories, and libraries
set(_sources
  src/ActionInitialization.cc
  src/Celeritas.cc
  src/DetectorConstruction.cc
  src/EventAction.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SensitiveDetector.cc
)

set(_includes
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Geant4_INCLUDE_DIR}
  ${Celeritas_ROOT}/include
)

set(_libs
  ${Geant4_LIBRARIES}
  Celeritas::celeritas
  Celeritas::accel
)

#-----------------------------------------------------------------------------#
# Add executable
add_executable(run-offload main.cc ${_sources})
target_include_directories(run-offload PRIVATE $<BUILD_INTERFACE:${_includes}>)

# Use Celeritas CUDA RDC utility function to correctly link libraries
include(CudaRdcUtils)
cuda_rdc_target_link_libraries(run-offload ${_libs})
