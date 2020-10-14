#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

# Note: 3.13 is the first CMake version that natively supports CUDA+MPI
cmake_minimum_required(VERSION 3.12)
project(Celeritas VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.18)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(CMAKE_VERSION VERSION_LESS 3.18)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/backport/3.18")
endif()

include(GNUInstallDirs)
include(CeleritasUtils)

#---------------------------------------------------------------------------##
# OPTIONS
#---------------------------------------------------------------------------##

# Components
option(CELERITAS_BUILD_DEMOS "Build Celeritas demonstration mini-apps" ON)
option(CELERITAS_BUILD_TESTS "Build Celeritas unit tests" ON)

# TPLs
option(CELERITAS_USE_CUDA "Enable GPU transport" ON)
option(CELERITAS_USE_Geant4 "Enable Geant4 adapter tools" OFF)
option(CELERITAS_USE_HepMC3 "Enable HepMC3 event record reader" OFF)
option(CELERITAS_USE_MPI "Enable distributed memory parallelism" ON)
option(CELERITAS_USE_ROOT "Enable ROOT I/O" OFF)
option(CELERITAS_USE_VecGeom "Enable VecGeom geometry" ON)

if(CELERITAS_USE_MPI AND CELERITAS_USE_CUDA AND CMAKE_VERSION VERSION_LESS 3.13)
  message(FATAL_ERROR "Celeritas requires CMake 3.13 or higher "
    "when building with CUDA + MPI.")
endif()

# Library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE STRING
      "Inform installed binaries of external library rpaths")
endif()
if(BUILD_SHARED_LIBS)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}" CACHE STRING
      "Inform installed binaries of internal library rpaths")
endif()
if(APPLE)
  option(CMAKE_MACOSX_RPATH "Support @rpath in install targets" ON)
endif()

# Build flags
option(CELERITAS_DEBUG "Enable runtime assertions" ON)
if(NOT CMAKE_BUILD_TYPE AND (CMAKE_GENERATOR STREQUAL "Ninja"
    OR CMAKE_GENERATOR STREQUAL "Unix Makefiles"))
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "Build type set to default by Celeritas CMakeLists" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

#---------------------------------------------------------------------------##
# DEPENDENCIES
#---------------------------------------------------------------------------##

if(CELERITAS_USE_CUDA)
  # Use host compiler by default to ensure ABI consistency
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING
    "Set to CMAKE_CXX_COMPILER by Celeritas CMakeLists")
  # Default to building device debug code
  set(CMAKE_CUDA_FLAGS_DEBUG "-g -G" CACHE STRING
    "Set by Celeritas CMakeLists")

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  find_package(CUDAToolkit REQUIRED QUIET)
endif()

if(CELERITAS_USE_Geant4)
  find_package(Geant4 REQUIRED)
endif()

if(CELERITAS_USE_HepMC3)
  find_package(HepMC3 REQUIRED)
endif()

if(CELERITAS_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(CELERITAS_USE_ROOT)
  find_package(ROOT REQUIRED)
endif()

if(CELERITAS_USE_VecGeom)
  find_package(VecGeom REQUIRED) # Minimum 1.1.7

  if((CELERITAS_USE_CUDA AND NOT VecGeom_CUDA_FOUND)
      OR (NOT CELERITAS_USE_CUDA AND VecGeom_CUDA_FOUND))
    message(SEND_ERROR "CUDA mismatch between the VecGeom installation "
      "at ${VECGEOM_INSTALL_DIR} (VecGeom_CUDA_FOUND=${VecGeom_CUDA_FOUND}) "
      "and Celeritas (CELERITAS_USE_CUDA=${CELERITAS_USE_CUDA})"
    )
  endif()
  if(NOT VecGeom_GDML_FOUND)
    message(SEND_ERROR "VecGeom GDML capability is required for Celeritas")
  endif()
endif()

if(CELERITAS_BUILD_TESTS AND CELERITAS_BUILD_DEMOS)
  find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)
endif()

#---------------------------------------------------------------------------##
# EXTERNALS
#---------------------------------------------------------------------------##
include(CeleritasLoadSubmodule)

if(NOT CELERITAS_GIT_SUBMODULE)
  set(_required_when_no_git REQUIRED)
else()
  set(_required_when_no_git QUIET)
endif()

if(CELERITAS_BUILD_DEMOS)
  find_package(nlohmann_json 3.7.0 ${_required_when_no_git})
endif()

if(CELERITAS_BUILD_TESTS)
  find_package(GTest ${_required_when_no_git})
endif()

add_subdirectory(external)

#---------------------------------------------------------------------------##
# LIBRARY
#---------------------------------------------------------------------------##

add_subdirectory(src)

#---------------------------------------------------------------------------##
# TESTS
#---------------------------------------------------------------------------##

if(CELERITAS_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()

#---------------------------------------------------------------------------##
# DEMO/HELPER APPS
#---------------------------------------------------------------------------##

add_subdirectory(app)

#---------------------------------------------------------------------------##
