#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CgvFindVersion.cmake")
cgv_find_version(Celeritas)
project(Celeritas VERSION "${Celeritas_VERSION}" LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.18)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(CMAKE_VERSION VERSION_LESS 3.18)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/backport/3.18")
endif()

include(GNUInstallDirs)
include(CeleritasUtils)

#----------------------------------------------------------------------------#
# OPTIONS
#----------------------------------------------------------------------------#

# Components
option(CELERITAS_BUILD_DEMOS "Build Celeritas demonstration mini-apps" ON)
option(CELERITAS_BUILD_DOCS  "Build Celeritas documentation" OFF)
option(CELERITAS_BUILD_TESTS "Build Celeritas unit tests" ON)

# TPLs
option(CELERITAS_USE_CUDA "Enable GPU transport" ON)
option(CELERITAS_USE_Geant4 "Enable Geant4 adapter tools" OFF)
option(CELERITAS_USE_HepMC3 "Enable HepMC3 event record reader" OFF)
option(CELERITAS_USE_JSON "Enable JSON I/O" "${CELERITAS_BUILD_DEMOS}")
option(CELERITAS_USE_MPI "Enable distributed memory parallelism" ON)
option(CELERITAS_USE_OpenMP "Enable CPU shared-memory parallelism" ON)
option(CELERITAS_USE_ROOT "Enable ROOT I/O" OFF)
option(CELERITAS_USE_SWIG_Python "Enable SWIG Python bindings" OFF)
option(CELERITAS_USE_VecGeom "Enable VecGeom geometry" ON)

if(CMAKE_VERSION VERSION_LESS 3.13 AND CELERITAS_USE_CUDA AND CELERITAS_USE_MPI)
  message(FATAL_ERROR "Celeritas requires CMake 3.13 or higher "
    "when building with CUDA + MPI.")
endif()
if(CMAKE_VERSION VERSION_LESS 3.17 AND CELERITAS_USE_CUDA
    AND CELERITAS_USE_VecGeom)
  message(FATAL_ERROR "VecGeom+CUDA has mysterious runtime errors under CMake "
    "3.17. Please, for the sake of your own mental health, "
    "use a newer version of CMake.")
endif()
if(CMAKE_VERSION VERSION_LESS 3.18 AND CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "The CMAKE_CUDA_ARCHITECTURES flag is not compatible "
    "with this version of CMake. Set CMAKE_CUDA_FLAGS.")
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

#----------------------------------------------------------------------------#
# DEPENDENCIES
#----------------------------------------------------------------------------#

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
  celeritas_find_package_config(Geant4 REQUIRED)
endif()

if(CELERITAS_USE_HepMC3)
  celeritas_find_package_config(HepMC3 REQUIRED)
  # HepMC3's cmake config file does not provide an imported target yet
  # Anticipate that it will, and create if not present. HepMC3 is
  # a private dep of Celeritas, so no need to re-export this.
  if(NOT TARGET HepMC3::HepMC3)
    add_library(HepMC3::HepMC3 UNKNOWN IMPORTED)
    set_target_properties(HepMC3::HepMC3 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${HEPMC3_INCLUDE_DIR}"
      IMPORTED_LOCATION "${HEPMC3_LIB}"
    )
  endif()
endif()

if(CELERITAS_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(CELERITAS_USE_ROOT)
  celeritas_find_package_config(ROOT REQUIRED)
endif()

if(CELERITAS_USE_SWIG_Python)
  if(CMAKE_VERSION VERSION_LESS 3.15)
    message(FATAL_ERROR "Celeritas requires CMake 3.15 or higher "
      "when building with SWIG Python.")
  endif()
  find_package(SWIG 4.0 REQUIRED)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
endif()

if(CELERITAS_USE_VecGeom)
  celeritas_find_package_config(VecGeom 1.1.13 REQUIRED)

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

  set_target_properties(VecGeom::vecgeom PROPERTIES
    CELERITAS_CUDA_STATIC_LIBRARY VecGeom::vecgeomcuda_static
    CELERITAS_CUDA_MIDDLE_LIBRARY VecGeom::vecgeomcuda
    CELERITAS_CUDA_FINAL_LIBRARY VecGeom::vecgeomcuda
  )
  set_target_properties(VecGeom::vecgeomcuda PROPERTIES
    CELERITAS_CUDA_LIBRARY_TYPE Shared
    CELERITAS_CUDA_STATIC_LIBRARY VecGeom::vecgeomcuda_static
    CELERITAS_CUDA_MIDDLE_LIBRARY VecGeom::vecgeomcuda
    CELERITAS_CUDA_FINAL_LIBRARY VecGeom::vecgeomcuda
  )
  set_target_properties(VecGeom::vecgeomcuda_static PROPERTIES
    CELERITAS_CUDA_LIBRARY_TYPE Static
    CELERITAS_CUDA_STATIC_LIBRARY VecGeom::vecgeomcuda_static
    CELERITAS_CUDA_MIDDLE_LIBRARY VecGeom::vecgeomcuda
    CELERITAS_CUDA_FINAL_LIBRARY VecGeom::vecgeomcuda
  )
endif()

if(NOT PYTHON_FOUND)
  # Python is required to launch tests for the demo apps *and* to regenerate
  # on-the-fly kernel launchers.
  unset(_required)
  if(CELERITAS_BUILD_TESTS AND CELERITAS_BUILD_DEMOS)
    set(_required REQUIRED)
  endif()
  find_package(Python 3.6 COMPONENTS Interpreter ${_required})
  unset(_required)
endif()

if(CELERITAS_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
endif()

#----------------------------------------------------------------------------#
# EXTERNALS
#----------------------------------------------------------------------------#
include(CeleritasLoadSubmodule)

if(NOT CELERITAS_GIT_SUBMODULE)
  set(_required_when_no_git REQUIRED)
else()
  set(_required_when_no_git QUIET)
endif()

if(CELERITAS_USE_JSON)
  find_package(nlohmann_json 3.7.0 ${_required_when_no_git})
endif()

if(CELERITAS_BUILD_TESTS)
  find_package(GTest ${_required_when_no_git})
endif()

add_subdirectory(external)

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

add_subdirectory(src)

#----------------------------------------------------------------------------#
# SWIG INTERFACE
#----------------------------------------------------------------------------#

if(CELERITAS_USE_SWIG_Python)
  add_subdirectory(interface)
endif()

#----------------------------------------------------------------------------#
# TESTS
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_TESTS)
  include(CTest)
  add_subdirectory(test)
endif()

#----------------------------------------------------------------------------#
# DEMO/HELPER APPS
#----------------------------------------------------------------------------#

add_subdirectory(app)

#----------------------------------------------------------------------------#
# DOCUMENTATION
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DOCS)
  add_subdirectory(doc)
endif()

#----------------------------------------------------------------------------#
