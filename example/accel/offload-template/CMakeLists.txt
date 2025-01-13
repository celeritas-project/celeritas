#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# Minimal Geant4 application with Celeritas offloading capabilities
#-----------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.12...3.30)
project(offload-template)

# Export/set Celeritas_ROOT environment variable pointing to its install dir
# Defaults to celeritas/install
set(ENV{Celeritas_ROOT} "../../../../install")

#-----------------------------------------------------------------------------#
# Find packages
find_package(Geant4 REQUIRED)
find_package(Celeritas 0.5 REQUIRED)

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
  src/TrackingAction.cc
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
add_executable(main main.cc ${_sources})
target_include_directories(main PRIVATE $<BUILD_INTERFACE:${_includes}>)

# Use Celeritas CUDA RDC utility function to correctly link libraries
include(CudaRdcUtils)
cuda_rdc_target_link_libraries(main ${_libs})
