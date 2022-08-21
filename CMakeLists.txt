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
include(CeleritasLibrary)
include(CeleritasUtils)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

#----------------------------------------------------------------------------#
# OPTIONS
#----------------------------------------------------------------------------#

celeritas_optional_language(CUDA)
if(NOT CELERITAS_USE_CUDA)
  celeritas_optional_language(HIP)
endif()

# Dependencies
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
option(CELERITAS_BUILD_DOCS "Build Celeritas documentation" OFF)
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

if(CMAKE_VERSION VERSION_LESS 3.22 AND CELERITAS_USE_HIP)
  message(WARNING "HIP support is immature; CMake 3.22+ is recommended.")
endif()
if(CELERITAS_USE_HIP AND CELERITAS_USE_VecGeom)
  message(SEND_ERROR "HIP is incompatible with VecGeom.")
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
  set(CMAKE_BUILD_TYPE "${_default_build_type}" CACHE STRING "Build type" FORCE)
  message(STATUS "Set default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  unset(_default_build_type)
endif()

# RNG selection
set(CELERITAS_RNG_OPTIONS XORWOW)
if(CELERITAS_USE_CUDA)
  list(APPEND CELERITAS_RNG_OPTIONS CURAND)
elseif(CELERITAS_USE_HIP)
  list(APPEND CELERITAS_RNG_OPTIONS HIPRAND)
elseif(NOT (CELERITAS_USE_CUDA OR CELERITAS_USE_HIP))
  # TODO: allow wrapper to standard library RNG when not building for device?
  # TODO: maybe even add wrapper to Geant4 RNG??
endif()
list(GET CELERITAS_RNG_OPTIONS 0 _DEFAULT_RNG)
set(CELERITAS_RNG "${_DEFAULT_RNG}" CACHE STRING
  "Celeritas runtime random number generator")
set_property(CACHE CELERITAS_RNG PROPERTY STRINGS "${CELERITAS_RNG_OPTIONS}")
list(FIND CELERITAS_RNG_OPTIONS "${CELERITAS_RNG}" _rng_index)
if(_rng_index EQUAL -1)
  message(SEND_ERROR "Invalid value CELERITAS_RNG=${CELERITAS_RNG}: must be "
    "${CELERITAS_RNG_OPTIONS}; overriding for next configure")
  set(CELERITAS_RNG "${_DEFAULT_RNG}" CACHE STRING
    "Celeritas runtime random number generator" FORCE)
endif()

cmake_dependent_option(CELERITAS_LAUNCH_BOUNDS
  "Use kernel launch bounds generated from launch-bounds.json" "OFF"
  "CELERITAS_USE_CUDA OR CELERITAS_USE_HIP" OFF
)

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
elseif(CELERITAS_USE_HIP)
  enable_language(HIP)
endif()

if(CELERITAS_USE_Geant4 AND NOT Geant4_FOUND)
  # Geant4 calls `include_directories` for CLHEP :( which is not what we want!
  # Save and restore include directories around the call -- even though as a
  # standalone project Celeritas will never have directory-level includes
  get_directory_property(_include_dirs INCLUDE_DIRECTORIES)
  find_package(Geant4 REQUIRED)
  set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${_include_dirs}")
endif()

if(CELERITAS_USE_HepMC3 AND NOT HepMC3_FOUND)
  find_package(HepMC3 REQUIRED)
endif()

if(CELERITAS_USE_JSON AND NOT nlohmann_json_FOUND)
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
  set(CELERITAS_PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING
    "Python path used for finding modules and generating documentation"
  )
endif()

if(CELERITAS_USE_ROOT)
  find_package(ROOT REQUIRED)
endif()

if(CELERITAS_USE_SWIG AND NOT SWIG_FOUND)
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

if(CELERITAS_BUILD_TESTS AND NOT GTest_FOUND)
  # TODO: download and build GTest as a subproject if not available
  find_package(GTest)
  if(NOT GTest_FOUND)
    message(SEND_ERROR "Googletest (GTest) is required for testing: setting "
      "CELERITAS_BUILD_TESTS=OFF for next build")
    set(CELERITAS_BUILD_TESTS OFF CACHE BOOL
      "Disabled because of missing GTest" FORCE)
  endif()
endif()

if(CELERITAS_BUILD_DOCS)
  if(NOT Doxygen_FOUND)
    find_package(Doxygen REQUIRED)
  endif()

  if(Python_FOUND)
    celeritas_check_python_module(CELERITAS_USE_Sphinx sphinx)
  endif()
  set(Sphinx_FOUND ${CELERITAS_USE_Sphinx})
  if(Sphinx_FOUND)
    celeritas_check_python_module(CELERITAS_USE_Breathe breathe)
    celeritas_check_python_module(CELERITAS_USE_SphinxBibtex "sphinxcontrib.bibtex")
    find_program(LATEXMK_EXECUTABLE latexmk)
  endif()
  if(NOT Sphinx_FOUND AND NOT Doxygen_FOUND)
    message(SEND_ERROR "Sphinx and/or Doxygen are required for testing: setting "
      "CELERITAS_BUILD_TESTS=OFF for next build")
    set(CELERITAS_BUILD_DOCS OFF CACHE BOOL
      "Disabled because of missing Doxygen/Sphinx" FORCE)
  endif()
endif()

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

# Define an interface library for propagating include paths for GPU compilers.
# This allow API calls, so long as the compiler or linker implicitly bring in
# the correct CUDA/ROCM libraries. Those libraries are *not* explicitly linked
# because mistakenly linking against both cudart and cudart_static (one of which
# might come from upstream libraries such as vecgeom) can cause nasty link- and
# run-time errors.
add_library(celeritas_device_toolkit INTERFACE)
add_library(Celeritas::DeviceToolkit ALIAS celeritas_device_toolkit)

if(CELERITAS_USE_CUDA)
  target_include_directories(celeritas_device_toolkit
    SYSTEM INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:${CUDAToolkit_INCLUDE_DIRS}>
  )
elseif(CELERITAS_USE_HIP)
  if(CMAKE_HIP_COMPILER_ROCM_ROOT)
    # Undocumented CMake variable
    set(ROCM_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
  else()
    # This hack works on Crusher as of ROCm 5.1.0
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm headers")
  endif()
  target_include_directories(celeritas_device_toolkit
    SYSTEM INTERFACE "${ROCM_PATH}/include"
  )
endif()

# Add the main libraries
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

if(CELERITAS_BUILD_DOCS)
  include(ExternalProject)
  add_subdirectory(doc)
endif()

#----------------------------------------------------------------------------#
# CONFIG FILE INSTALLATION
#----------------------------------------------------------------------------#

# Where to configure cmake files
set(CELER_CMAKE_CONFIG_DIRECTORY "${PROJECT_BINARY_DIR}/lib/cmake")
# Where to install configured cmake files
set(CELER_INSTALL_CMAKECONFIGDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/Celeritas")

# Build list of CMake files to install
set(_cmake_files
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasGen.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasGen/gen-action.py"
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasLibrary.cmake"
)
foreach(_dep Geant4 HepMC3 ROOT VecGeom)
  if(CELERITAS_USE_${_dep})
    list(APPEND _cmake_files "${PROJECT_SOURCE_DIR}/cmake/Find${_dep}.cmake")
  endif()
endforeach()
if(CELERITAS_BUILD_TESTS)
  list(APPEND _cmake_files
    "${PROJECT_SOURCE_DIR}/cmake/CeleritasAddTest.cmake"
  )
endif()

install(FILES ${_cmake_files}
  DESTINATION "${CELER_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/backport"
  DESTINATION "${CELER_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)

# Export all cache variables that start with CELERITAS_
set(CELER_EXPORT_VARIABLES)
get_directory_property(_cachevar_keys CACHE_VARIABLES)
foreach(_key IN LISTS _cachevar_keys)
  if(_key MATCHES "^CELERITAS_")
    list(APPEND CELER_EXPORT_VARIABLES "set(${_key} \"${${_key}}\")")
  endif()
endforeach()

# Add other cache variables, prefixed with CELER_
foreach(_key BUILD_SHARED_LIBS)
  list(APPEND CELER_EXPORT_VARIABLES "set(CELERITAS_${_key} \"${${_key}}\")")
endforeach()

# Add hints for direct dependencies and indirect geant dependencies
list(APPEND CELER_EXPORT_VARIABLES "# Hints for upstream dependencies")
foreach(_key
  MPIEXEC_EXECUTABLE CUDAToolkit_BIN_DIR
  Geant4_DIR HepMC3_DIR nlohmann_json_DIR Python_DIR ROOT_DIR
  VecCore_DIR VecGeom_DIR
  CLHEP_DIR ZLIB_DIR EXPAT_DIR XercesC_DIR PTL_DIR
  EXPAT_INCLUDE_DIR EXPAT_LIBRARY
  XercesC_LIBRARY XercesC_INCLUDE_DIR
)
  set(_val "${${_key}}")
  if(_val)
    list(APPEND CELER_EXPORT_VARIABLES
      "if(NOT DEFINED ${_key})"
      "  set(${_key} \"${_val}\")"
      "endif()"
    )
  endif()
endforeach()
list(JOIN CELER_EXPORT_VARIABLES "\n" CELER_EXPORT_VARIABLES)

# Generate the file needed by downstream "find_package(CELER)"
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasConfig.cmake.in"
  "${CELER_CMAKE_CONFIG_DIRECTORY}/CeleritasConfig.cmake"
  @ONLY
)

# Export version info
write_basic_package_version_file(
  "${CELER_CMAKE_CONFIG_DIRECTORY}/CeleritasConfigVersion.cmake"
  COMPATIBILITY ExactVersion
)

# Install the config and version files
install(FILES
  "${CELER_CMAKE_CONFIG_DIRECTORY}/CeleritasConfig.cmake"
  "${CELER_CMAKE_CONFIG_DIRECTORY}/CeleritasConfigVersion.cmake"
  DESTINATION ${CELER_INSTALL_CMAKECONFIGDIR}
  COMPONENT development
)

# Install 'CeleritasTargets.cmake', included by CeleritasConfig.cmake, which
# references the targets we install.
install(EXPORT celeritas-targets
  FILE CeleritasTargets.cmake
  NAMESPACE Celeritas::
  DESTINATION "${CELER_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)

#----------------------------------------------------------------------------#
