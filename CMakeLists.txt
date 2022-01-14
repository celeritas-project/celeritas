#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/CgvFindVersion.cmake")
cgv_find_version(Celeritas)
set(CMAKE_USER_MAKE_RULES_OVERRIDE
  "${CMAKE_CURRENT_LIST_DIR}/cmake/CeleritasMakeRulesOverride.cmake")

project(Celeritas VERSION "${Celeritas_VERSION}" LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(CMAKE_VERSION VERSION_LESS 3.18)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/backport/3.18")
endif()

include(GNUInstallDirs)
include(CeleritasUtils)

#----------------------------------------------------------------------------#
# OPTIONS
#----------------------------------------------------------------------------#

celeritas_optional_language(CUDA)

# Dependencies
celeritas_optional_package(Doxygen "Generate code documentation")
celeritas_optional_package(Geant4 "Enable Geant4 adapter tools")
celeritas_optional_package(HepMC3 "Enable HepMC3 event record reader")
celeritas_optional_package(JSON nlohmann_json "Enable JSON I/O")
celeritas_optional_package(MPI "Enable distributed memory parallelism")
celeritas_optional_package(OpenMP "Enable CPU shared-memory parallelism")
celeritas_optional_package(Python "Use Python to generate and preprocess")
celeritas_optional_package(ROOT "Enable ROOT I/O")
celeritas_optional_package(VecGeom "Use VecGeom geometry")

if(CELERITAS_USE_Python)
  celeritas_optional_package(SWIG "Build SWIG Python bindings")
endif()

# Components
option(CELERITAS_BUILD_DEMOS "Build Celeritas demonstration mini-apps"
  ${CELERITAS_USE_JSON})
option(CELERITAS_BUILD_TESTS "Build Celeritas unit tests" ON)

if(CMAKE_VERSION VERSION_LESS 3.13 AND CELERITAS_USE_CUDA AND CELERITAS_USE_MPI)
  message(FATAL_ERROR "Celeritas requires CMake 3.13 or higher "
    "when building with CUDA + MPI.")
endif()
if(CMAKE_VERSION VERSION_LESS 3.18 AND CELERITAS_USE_CUDA
    AND CELERITAS_USE_VecGeom)
  message(FATAL_ERROR "VecGeom+CUDA requires CMake 3.18 or higher to support "
    "\"Separable compilation\".")
endif()
if(CMAKE_VERSION VERSION_LESS 3.18 AND CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "The CMAKE_CUDA_ARCHITECTURES flag is not compatible "
    "with this version of CMake. Set CMAKE_CUDA_FLAGS.")
endif()

# Library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
  # Inform installed binaries of external library rpaths
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()
if(BUILD_SHARED_LIBS AND NOT DEFINED CMAKE_INSTALL_RPATH)
  # Inform installed binaries of internal library rpaths
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()

# Build flags
option(CELERITAS_DEBUG "Enable runtime assertions" ON)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE)
  if(CELERITAS_DEBUG)
    set(_default_build_type "Debug")
  else()
    set(_default_build_type "Release")
  endif()
  set(CMAKE_BUILD_TYPE "${_default_build_type}" CACHE STRING
    "Build type [None Debug Release RelWithDebInfo MinSizeRel]"
    FORCE)
  message(STATUS "Set default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  unset(_default_build_type)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  # Default CMake standard to C++14 for this directory and underneath
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

#----------------------------------------------------------------------------#
# DEPENDENCIES
#----------------------------------------------------------------------------#

if(CELERITAS_USE_CUDA)
  # Use host compiler by default to ensure ABI consistency
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING
    "Set to CMAKE_CXX_COMPILER by Celeritas CMakeLists")

  enable_language(CUDA)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_EXTENSIONS ${CMAKE_CXX_EXTENSIONS})
  endif()
  unset(CMAKE_CUDA_FLAGS_DEBUG_INIT)

  find_package(CUDAToolkit REQUIRED QUIET)
endif()

if(CELERITAS_USE_Doxygen)
  find_package(Doxygen REQUIRED)
endif()

if(CELERITAS_USE_Geant4)
  find_package(Geant4 REQUIRED)
endif()

if(CELERITAS_USE_HepMC3)
  find_package(HepMC3 REQUIRED)
endif()

if(CELERITAS_USE_JSON)
  find_package(nlohmann_json 3.7.0 REQUIRED)
endif()

if(CELERITAS_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(CELERITAS_USE_OpenMP)
  find_package(OpenMP REQUIRED)
endif()

if(CELERITAS_USE_Python)
  set(_components Interpreter)
  if(CELERITAS_USE_SWIG)
    list(APPEND _components Development)
  endif()
  find_package(Python 3.6 REQUIRED COMPONENTS ${_components})
endif()

if(CELERITAS_USE_ROOT)
  find_package(ROOT REQUIRED)
endif()

if(CELERITAS_USE_SWIG)
  find_package(SWIG 4.0 REQUIRED)
endif()

if(CELERITAS_USE_VecGeom)
  find_package(VecGeom 1.1.17 REQUIRED)

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

if(CELERITAS_BUILD_TESTS)
  # TODO: download and build GTest as a subproject if not available
  find_package(GTest)
  if(NOT GTest_FOUND)
    message(SEND_ERROR "Googletest (GTest) is required for testing: setting "
      "CELERITAS_BUILD_TESTS=OFF for next build")
    set(CELERITAS_BUILD_TESTS OFF CACHE BOOL
      "Disabled because of missing GTest" FORCE)
  endif()
endif()

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

add_subdirectory(src)

#----------------------------------------------------------------------------#
# SWIG INTERFACE
#----------------------------------------------------------------------------#

if(CELERITAS_USE_SWIG)
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

if(CELERITAS_USE_Doxygen)
  add_subdirectory(doc)
endif()

#----------------------------------------------------------------------------#
