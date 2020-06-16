#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

# Note: 3.13 is the first CMake version that natively supports CUDA+MPI
cmake_minimum_required(VERSION 3.13)
project(Celeritas VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.13...3.16)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(CMAKE_VERSION VERSION_LESS 3.18)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/backport/3.18")
endif()

include(GNUInstallDirs)
include(CeleritasUtils)

#---------------------------------------------------------------------------##
# OPTIONS
#---------------------------------------------------------------------------##

# Tests
option(CELERITAS_BUILD_TESTS "Build Celeritas unit tests" ON)

# TPLs
option(CELERITAS_USE_CUDA "Enable GPU transport" ON)
option(CELERITAS_USE_MPI "Enable distributed memory parallelism" ON)
option(CELERITAS_USE_ROOT "Enable ROOT I/O" OFF)
option(CELERITAS_USE_VECGEOM "Enable VecGeom geometry" ON)

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
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

#---------------------------------------------------------------------------##
# DEPENDENCIES
#---------------------------------------------------------------------------##

if(CELERITAS_USE_CUDA)
  # Use host compiler by default to ensure ABI consistency
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "")
  # Default to building device debug code
  set(CMAKE_CUDA_FLAGS_DEBUG "-g -G" CACHE STRING "")

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  find_package(CUDAToolkit REQUIRED QUIET)
endif()

if(CELERITAS_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(CELERITAS_USE_ROOT)
  find_package(ROOT REQUIRED)
endif()

if(CELERITAS_USE_VECGEOM)
  find_package(VecGeom REQUIRED)
  # 1.1.7 is required, but the VecGeomConfig.cmake doesn't supply version
  if(NOT TARGET VecGeom::vecgeom)
    message(SEND_ERROR "The version of VecGeom at ${VECGEOM_INSTALL_DIR} "
      "does not define the expected library targets")
  endif()

  if(NOT VECGEOM_CUDA_ARCH STREQUAL "none")
    set(_vecgeom_cuda TRUE)
  endif()
  if((CELERITAS_USE_CUDA AND NOT VECGEOM_CUDA_ARCH)
      OR (NOT CELERITAS_USE_CUDA AND VECGEOM_CUDA_ARCH))
    message(SEND_ERROR "CUDA mismatch between the VecGeom installation "
      "at ${VECGEOM_INSTALL_DIR} (VECGEOM_CUDA_ARCH=${VECGEOM_CUDA_ARCH}) "
      "and Celeritas (CELERITAS_USE_CUDA=${CELERITAS_USE_CUDA})"
    )
  endif()
  set(_vecgeom_cuda)
  if(NOT VecGeom_GDML_FOUND)
    message(SEND_ERROR "VecGeom GDML capability is required for Celeritas")
  endif()
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
# DEMO APP
#---------------------------------------------------------------------------##

# add_subdirectory(app)

#---------------------------------------------------------------------------##
