#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)

# Set Celeritas_VERSION using git tags using the following format
set(CGV_TAG_REGEX "v([0-9.]+)(-dev|-rc.[0-9]+)?")
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
# MAIN OPTIONS
#----------------------------------------------------------------------------#

# NOTE: languages must be *first* because their settings affect the
# find_package calls.
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

# Assertion handling
option(CELERITAS_DEBUG "Enable runtime assertions" ON)

# Secondary testing options (only applies to CTest)
if(NOT CELERITAS_DEBUG OR CELERITAS_USE_VecGeom)
  set(_default_lock ON)
else()
  set(_default_lock OFF)
endif()
cmake_dependent_option(CELERITAS_TEST_RESOURCE_LOCK
  "Only run one GPU-enabled test at a time" "${_default_lock}"
  "CELERITAS_BUILD_TESTS" OFF
)
cmake_dependent_option(CELERITAS_TEST_VERBOSE
  "Increase logging level for tests" "${CELERITAS_DEBUG}"
  "CELERITAS_BUILD_TESTS" OFF
)

#----------------------------------------------------------------------------#
# CELERITAS CORE IMPLEMENTATION OPTIONS
#----------------------------------------------------------------------------#

# CELERITAS_CORE_RNG: random number generator selection
celeritas_setup_option(CELERITAS_CORE_RNG xorwow)
celeritas_setup_option(CELERITAS_CORE_RNG cuRAND CELERITAS_USE_CUDA)
celeritas_setup_option(CELERITAS_CORE_RNG hipRAND CELERITAS_USE_HIP)
# TODO: allow wrapper to standard library RNG when not building for device?
# TODO: maybe even add wrapper to Geant4 RNG??
celeritas_define_options(CELERITAS_CORE_RNG
  "Celeritas runtime random number generator")

# CELERITAS_CORE_GEO: runtime geometry selection
if(CELERITAS_USE_VecGeom AND NOT CELERITAS_USE_HIP)
  set(_allow_vecgeom TRUE)
else()
  set(_allow_vecgeom FALSE)
  if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
    message(SEND_ERROR "VecGeom core geometry is incompatible with HIP")
  endif()
endif()
if(CELERITAS_USE_Geant4 AND NOT (CELERITAS_USE_HIP OR CELERITAS_USE_CUDA OR CELERITAS_USE_OpenMP))
  set(_allow_g4 TRUE)
else()
  if(CELERITAS_CORE_GEO STREQUAL "Geant4")
    message(SEND_ERROR "Geant4 core geometry is incompatible with HIP, CUDA, and OpenMP")
  endif()
  set(_allow_g4 FALSE)
endif()
celeritas_setup_option(CELERITAS_CORE_GEO VecGeom _allow_vecgeom)
celeritas_setup_option(CELERITAS_CORE_GEO ORANGE)
celeritas_setup_option(CELERITAS_CORE_GEO Geant4 _allow_g4)
celeritas_define_options(CELERITAS_CORE_GEO "Celeritas runtime geometry")

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(CELERITAS_MAX_BLOCK_SIZE 256
    CACHE STRING "Threads-per-block launch bound for Celeritas action kernels"
  )
else()
  set(CELERITAS_MAX_BLOCK_SIZE 0)
endif()

if(CELERITAS_CORE_GEO STREQUAL Geant4)
  set(_allow_single_prec FALSE)
else()
  set(_allow_single_prec TRUE)
endif()

celeritas_setup_option(CELERITAS_REAL_TYPE double)
celeritas_setup_option(CELERITAS_REAL_TYPE float)
celeritas_define_options(CELERITAS_REAL_TYPE
  "Global runtime precision for real numbers")

#----------------------------------------------------------------------------#
# CMAKE VERSION CHECKS
#----------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------#
# CMAKE INTRINSIC OPTIONS
#
# These are generally used to initialize properties on targets, and it's
# possible Celeritas is being built inside another project. Instead of saving
# these as cache variables (which change the defaults project-wide, including
# changing behavior of other code that was loaded before Celeritas) set them as
# local variables to be inherited underneath Celeritas.
#----------------------------------------------------------------------------#

### Configuration ###
celeritas_set_default(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
celeritas_set_default(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
celeritas_set_default(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)

### Build flags ###
# Default to building CTest tree if using Celeritas tests
celeritas_set_default(BUILD_TESTING ${CELERITAS_BUILD_TESTS})
# Default to debug or released based on value of CELERITAS_DEBUG
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
# Default to using C++17 everywhere
celeritas_set_default(CMAKE_CXX_STANDARD 17)
celeritas_set_default(CMAKE_CXX_EXTENSIONS OFF)
if(CELERITAS_USE_CUDA)
  # Default to setting CUDA C++ standard the same as C++
  celeritas_set_default(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  celeritas_set_default(CMAKE_CUDA_EXTENSIONS ${CMAKE_CXX_EXTENSIONS})
endif()

### Linking flags ###
# Default to building shared libraries (*not* a cache variable)
celeritas_set_default(BUILD_SHARED_LIBS ON)
# Inform installed binaries of external library rpaths
celeritas_set_default(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
if(BUILD_SHARED_LIBS)
  # Inform installed binaries of internal library rpaths
  celeritas_set_default(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
  # Do not relink libs/binaries when dependent shared libs change
  celeritas_set_default(CMAKE_LINK_DEPENDS_NO_SHARED ON)
endif()
if(BUILD_SHARED_LIBS OR CELERITAS_USE_ROOT)
  # Make sure modules, sub-libraries, etc. are relocatable
  celeritas_set_default(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

### Installation flags ###
# When developing add checking for proper usage of `install(`
if(CELERITAS_DEBUG)
  celeritas_set_default(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
endif()
# Avoid printing details about already installed files
celeritas_set_default(CMAKE_INSTALL_MESSAGE LAZY)

#----------------------------------------------------------------------------#
# Output locations for Celeritas products (used by CeleritasLibrary.cmake,
# CeleritasUtils.cmake, install code below) will mirror the installation layout
set(CELERITAS_CMAKE_CONFIG_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake")
set(CELERITAS_HEADER_CONFIG_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}")
set(CELERITAS_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CELERITAS_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

#----------------------------------------------------------------------------#
# DEPENDENCIES
#----------------------------------------------------------------------------#

if(BUILD_TESTING)
  # Build the test tree for unit and/or app tests
  include(CTest)
endif()

if(CELERITAS_USE_CUDA)
  # Use host compiler by default to ensure ABI consistency
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING
    "Set to CMAKE_CXX_COMPILER by Celeritas CMakeLists")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED QUIET)
elseif(CELERITAS_USE_HIP)
  enable_language(HIP)
  # NOTE: known broken up to 5.6; unknown after
  if(CELERITAS_DEBUG AND "${hip-lang_VERSION}" VERSION_LESS 5.7)
    celeritas_error_incompatible_option(
      "HIP cannot build debug code (unhandled SGPR spill to memory)"
      CELERITAS_DEBUG
      OFF
    )
  endif()
endif()

if(CELERITAS_USE_Geant4 AND NOT Geant4_FOUND)
  find_package(Geant4 REQUIRED)
endif()

if(CELERITAS_USE_HepMC3)
  if(NOT HepMC3_FOUND)
    find_package(HepMC3 REQUIRED)
  endif()
  set(HepMC3_LIBRARIES HepMC3::HepMC3)
endif()

if(CELERITAS_USE_JSON)
  if(NOT nlohmann_json_FOUND)
    find_package(nlohmann_json 3.7.0 REQUIRED)
  endif()
  set(nlohmann_json_LIBRARIES nlohmann_json::nlohmann_json)
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
  if(NOT Python_FOUND)
    find_package(Python 3.6 REQUIRED COMPONENTS ${_components})
  endif()
  set(CELERITAS_PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING
    "Python path used for finding modules and generating documentation"
  )
endif()

if(CELERITAS_USE_ROOT)
  if(NOT BUILD_SHARED_LIBS)
    celeritas_error_incompatible_option(
      "ROOT PCM fails at runtime without shared libraries"
      BUILD_SHARED_LIBS
      ON
    )
  endif()
  # Older ROOT versions are missing CMake macros
  find_package(ROOT 6.24 REQUIRED)
endif()

if(CELERITAS_USE_SWIG)
  if(CELERITAS_REAL_TYPE STREQUAL "float")
    celeritas_error_incompatible_option(
      "Single-precision runtime is not compatible with SWIG"
      CELERITAS_USE_SWIG
      OFF
    )
  endif()
  if(NOT SWIG_FOUND)
    # Note: 4.1 is required for C++17 parsing
    find_package(SWIG 4.1 REQUIRED)
  endif()
endif()

if(CELERITAS_USE_VecGeom)
  if(NOT VecGeom_FOUND)
    find_package(VecGeom 1.2.4 REQUIRED)
  endif()

  if(CELERITAS_USE_CUDA AND NOT VecGeom_CUDA_FOUND)
    celeritas_error_incompatible_option(
      "VecGeom installation at ${VECGEOM_INSTALL_DIR} is not CUDA-enabled"
      CELERITAS_USE_CUDA
      "${VecGeom_CUDA_FOUND}"
    )
  endif()
  if(CELERITAS_REAL_TYPE STREQUAL "float" AND NOT VecGeom_single_precision_FOUND)
    celeritas_error_incompatible_option(
      "VecGeom installation at ${VECGEOM_INSTALL_DIR} uses double precision"
      CELERITAS_REAL_TYPE
      "double"
    )
  endif()
  if(CELERITAS_REAL_TYPE STREQUAL "double" AND VecGeom_single_precision_FOUND)
    celeritas_error_incompatible_option(
      "VecGeom installation at ${VECGEOM_INSTALL_DIR} uses single precision"
      CELERITAS_REAL_TYPE
      "float"
    )
  endif()
  if(NOT VecGeom_GDML_FOUND)
    message(SEND_ERROR "VecGeom GDML capability is required for Celeritas")
  endif()
endif()

if(CELERITAS_BUILD_DEMOS AND NOT CELERITAS_USE_JSON)
  celeritas_error_incompatible_option(
    "JSON support is required for demos but was not found"
    CELERITAS_BUILD_DEMOS
    OFF
  )
endif()

if(CELERITAS_BUILD_DOCS)
  if(NOT Doxygen_FOUND)
    find_package(Doxygen)
  endif()
  if(NOT Doxygen_FOUND)
    celeritas_error_incompatible_option(
      "Doxygen is required for building documentation but was not found"
      CELERITAS_BUILD_DOCS
      OFF
    )
  endif()

  if(CELERITAS_USE_Python)
    celeritas_check_python_module(CELERITAS_USE_Sphinx sphinx)
  endif()
  set(Sphinx_FOUND ${CELERITAS_USE_Sphinx})
  if(Sphinx_FOUND)
    celeritas_check_python_module(CELERITAS_USE_Breathe breathe)
    celeritas_check_python_module(CELERITAS_USE_Furo "furo")
    celeritas_check_python_module(CELERITAS_USE_SphinxBibtex "sphinxcontrib.bibtex")
    if(CELERITAS_USE_Breathe)
      find_program(LATEXMK_EXECUTABLE latexmk)
    endif()
  endif()
endif()

if(CELERITAS_BUILD_TESTS)
  # TODO: download and build GTest as a subproject if not available
  if(NOT GTest_FOUND)
    find_package(GTest 1.10)
  endif()
  if(NOT GTest_FOUND)
    celeritas_error_incompatible_option(
      "Googletest (GTest) is required for testing but was not found"
      CELERITAS_BUILD_TESTS
      OFF
    )
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
  target_link_libraries(celeritas_device_toolkit INTERFACE CUDA::toolkit)
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
  find_library(ROCTX_LIBRARY roctx64)
  if(ROCTX_LIBRARY)
    set(CELERITAS_HAVE_ROCTX ON)
    target_link_libraries(celeritas_device_toolkit INTERFACE ${ROCTX_LIBRARY})
  endif()
  # TODO: downstream libs don't link against correct HIP dependencies when using
  # static libraries
endif()

install(TARGETS celeritas_device_toolkit
  EXPORT celeritas-targets
)

#----------------------------------------------------------------------------#
# Add the main libraries
add_subdirectory(src)

#----------------------------------------------------------------------------#
# SWIG INTERFACE
#----------------------------------------------------------------------------#

if(CELERITAS_USE_SWIG)
  add_subdirectory(interface)
endif()

#----------------------------------------------------------------------------#
# UNIT TESTS
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_TESTS)
  add_subdirectory(test)
endif()

#----------------------------------------------------------------------------#
# APPLICATIONS AND BINARIES
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

# Where to install configured cmake files
set(CELERITAS_INSTALL_CMAKECONFIGDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/Celeritas")

# Build list of CMake files to install
set(_cmake_files
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
  DESTINATION "${CELERITAS_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/backport"
  DESTINATION "${CELERITAS_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)

# Export all cache variables that start with CELERITAS_
set(CELERITAS_EXPORT_VARIABLES)
macro(celeritas_export_var varname)
  list(APPEND CELERITAS_EXPORT_VARIABLES "set(${varname} \"${${varname}}\")")
endmacro()
celeritas_export_var(Celeritas_VERSION_STRING)
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Configuration options")
get_directory_property(_cachevar_keys CACHE_VARIABLES)
foreach(_key IN LISTS _cachevar_keys)
  if(_key MATCHES "^CELERITAS_")
    celeritas_export_var(${_key})
  endif()
endforeach()

# Export defaulted variables
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Defaulted variables")
foreach(_key IN LISTS CELERITAS_DEFAULT_VARIABLES)
  list(APPEND CELERITAS_EXPORT_VARIABLES "set(CELERITAS_${_key} \"${${_key}}\")")
endforeach()

# Add hints for direct dependencies and indirect geant dependencies
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Hints for upstream dependencies")
foreach(_key
  MPIEXEC_EXECUTABLE CUDAToolkit_BIN_DIR
  Geant4_DIR GTest_DIR HepMC3_DIR nlohmann_json_DIR Python_DIR ROOT_DIR
  VecCore_DIR VecGeom_DIR
  CLHEP_DIR ZLIB_DIR EXPAT_DIR XercesC_DIR PTL_DIR
  EXPAT_INCLUDE_DIR EXPAT_LIBRARY
  XercesC_LIBRARY XercesC_INCLUDE_DIR
)
  set(_val "${${_key}}")
  if(_val)
    list(APPEND CELERITAS_EXPORT_VARIABLES
      "if(NOT DEFINED ${_key})"
      "  set(${_key} \"${_val}\")"
      "endif()"
    )
  endif()
endforeach()
list(JOIN CELERITAS_EXPORT_VARIABLES "\n" CELERITAS_EXPORT_VARIABLES)

# Generate the file needed by downstream "find_package(CELER)"
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasConfig.cmake.in"
  "${CELERITAS_CMAKE_CONFIG_DIRECTORY}/CeleritasConfig.cmake"
  @ONLY
)

# Export version info
# TODO for version 1.0.0: change COMPATIBILITY to MajorVersion
write_basic_package_version_file(
  "${CELERITAS_CMAKE_CONFIG_DIRECTORY}/CeleritasConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
)

# Install generated config files
install(DIRECTORY "${CELERITAS_HEADER_CONFIG_DIRECTORY}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
  FILES_MATCHING REGEX ".*\\.hh?$"
)

# Install the config and version files
install(FILES
  "${CELERITAS_CMAKE_CONFIG_DIRECTORY}/CeleritasConfig.cmake"
  "${CELERITAS_CMAKE_CONFIG_DIRECTORY}/CeleritasConfigVersion.cmake"
  DESTINATION ${CELERITAS_INSTALL_CMAKECONFIGDIR}
  COMPONENT development
)

# Install 'CeleritasTargets.cmake', included by CeleritasConfig.cmake, which
# references the targets we install.
install(EXPORT celeritas-targets
  FILE CeleritasTargets.cmake
  NAMESPACE Celeritas::
  DESTINATION "${CELERITAS_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)

if(Celeritas_VERSION VERSION_EQUAL "0.0.0")
  install(CODE "
message(WARNING \"The Celeritas version was not detected during configuration.
  (Check the beginning of your initial configure output for more details.)
  This will result in the installation having incorrect version metadata and
  will interfere with downstream CMake version requirements and may obscure
  provenance data in output results.\")
")
endif()


#----------------------------------------------------------------------------#
