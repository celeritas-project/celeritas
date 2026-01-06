#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18...4.1)

# Fallback version in case of shallow clone: update when tagging "-dev"
set(Celeritas_VERSION 0.7)
# Load actual version from git metadata
include("${CMAKE_CURRENT_LIST_DIR}/cmake/CgvFindVersion.cmake")
cgv_find_version(Celeritas)

project(Celeritas VERSION "${Celeritas_VERSION}" LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(GNUInstallDirs)
include(CeleritasOptionUtils)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

#----------------------------------------------------------------------------#
# PACKAGE OPTIONS
#----------------------------------------------------------------------------#

# NOTE: languages must be *first* because their settings affect the
# find_package calls.
celeritas_optional_language(CUDA)
if(NOT CELERITAS_USE_CUDA)
  celeritas_optional_language(HIP)
endif()

# Optional dependencies
celeritas_optional_package(covfie "Use Covfie field integrator")
celeritas_optional_package(Geant4 "Enable Geant4 adapter tools")
celeritas_optional_package(HepMC3 "Enable HepMC3 event record reader")
celeritas_optional_package(LArSoft "Build LArSoft data interfaces")
celeritas_optional_package(MPI "Enable distributed memory parallelism")
celeritas_optional_package(OpenMP "Enable CPU shared-memory parallelism")
celeritas_optional_package(PNG "Enable PNG output with libpng")
celeritas_optional_package(Python "Use Python for documentation and testing")
celeritas_optional_package(ROOT "Enable ROOT I/O")
celeritas_optional_package(VecGeom "Use VecGeom geometry")
celeritas_optional_package(DD4hep "Enable DD4hep integration")
option(CELERITAS_USE_Perfetto "Perfetto tracing library" OFF)

# Components
option(CELERITAS_BUILD_APPS "Build Celeritas executables (strongly recommended)" ON)
option(CELERITAS_BUILD_DOCS "Build Celeritas documentation" OFF)
option(CELERITAS_BUILD_TESTS "Build Celeritas unit tests" OFF)

# Assertion handling
option(CELERITAS_DEBUG "Enable runtime assertions" OFF)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_celeritas_use_device TRUE)
else()
  set(_celeritas_use_device FALSE)
endif()
cmake_dependent_option(CELERITAS_DEVICE_DEBUG
  "Use verbose debug assertions in device code" "OFF"
  "_celeritas_use_device;CELERITAS_DEBUG" OFF
)

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
if(CELERITAS_BUILD_TESTS)
  # NOTE: CMake "normalizes" this path by stripping trailing directory
  # separators, so this *must* be a directory.
  set(CELERITAS_TEST_XML "" CACHE PATH
    "If non-empty, write JUnit output from google tests to this directory"
  )
  mark_as_advanced(CELERITAS_TEST_XML)
endif()

# Automatic options
celeritas_force_package(CLI11 ${CELERITAS_BUILD_APPS})
celeritas_force_package(GTest ${CELERITAS_BUILD_TESTS})

#----------------------------------------------------------------------------#
# PACKAGE-SPECIFIC OPTIONS
#----------------------------------------------------------------------------#

if(CELERITAS_USE_Python)
  # Note that these are used by the celeritas_get_python_env command
  set(CELERITAS_PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING
    "Python path used for finding modules and generating documentation"
  )
  set(CELERITAS_PYTHONWARNINGS "$ENV{PYTHONWARNINGS}" CACHE STRING
    "Warning options to send to Python"
  )
endif()

#----------------------------------------------------------------------------#
# CELERITAS CORE IMPLEMENTATION OPTIONS
#----------------------------------------------------------------------------#

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(CELERITAS_MAX_BLOCK_SIZE 256
    CACHE STRING "Threads-per-block launch bound for Celeritas action kernels"
  )
else()
  set(CELERITAS_MAX_BLOCK_SIZE)
endif()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_CORE_RNG
# Random number generator selection
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
celeritas_setup_option(CELERITAS_CORE_RNG xorwow)
celeritas_setup_option(CELERITAS_CORE_RNG ranluxpp)
celeritas_setup_option(CELERITAS_CORE_RNG cuRAND CELERITAS_USE_CUDA)
celeritas_setup_option(CELERITAS_CORE_RNG hipRAND CELERITAS_USE_HIP)
# TODO: add wrapper to standard library RNG when not building for device?
# TODO: maybe even add wrapper to Geant4 RNG??
celeritas_define_options(CELERITAS_CORE_RNG
  "Celeritas runtime random number generator")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_CORE_GEO
# Runtime geometry selection
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if(CELERITAS_USE_VecGeom AND NOT CELERITAS_USE_HIP)
  set(_allow_vecgeom TRUE)
else()
  set(_allow_vecgeom FALSE)
  if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
    message(SEND_ERROR "VecGeom core geometry is incompatible with HIP")
  endif()
endif()
if(CELERITAS_USE_Geant4 AND NOT (CELERITAS_USE_HIP OR CELERITAS_USE_CUDA OR CELERITAS_OPENMP))
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_OPENMP
# Thread parallelism
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if(CELERITAS_USE_OpenMP)
  set(_disable_openmp FALSE)
else()
  set(_disable_openmp TRUE)
endif()

celeritas_setup_option(CELERITAS_OPENMP disabled _disable_openmp)
celeritas_setup_option(CELERITAS_OPENMP event CELERITAS_USE_OpenMP)
celeritas_setup_option(CELERITAS_OPENMP track CELERITAS_USE_OpenMP)
celeritas_define_options(CELERITAS_OPENMP "Celeritas OpenMP parallelism")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_UNITS
# Unit system for celeritas runtime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if(CELERITAS_CORE_GEO STREQUAL Geant4)
  set(_allow_single_prec FALSE)
else()
  set(_allow_single_prec TRUE)
endif()

celeritas_setup_option(CELERITAS_UNITS CGS)
celeritas_setup_option(CELERITAS_UNITS SI)
celeritas_setup_option(CELERITAS_UNITS CLHEP)
celeritas_define_options(CELERITAS_UNITS
  "Native unit system for Celeritas")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_CODATA
# Experimentally measured physical constants
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
celeritas_setup_option(CELERITAS_CODATA 2006) # Used by CLHEP
celeritas_setup_option(CELERITAS_CODATA 2018)
celeritas_setup_option(CELERITAS_CODATA 2022)
set(CELERITAS_CODATA_OPTION_DEFAULT 2018)
celeritas_define_options(CELERITAS_CODATA
  "Experimentally measured physical constants revision")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CELERITAS_REAL_TYPE
# Precision for real numbers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
celeritas_setup_option(CELERITAS_REAL_TYPE double)
celeritas_setup_option(CELERITAS_REAL_TYPE float)
celeritas_define_options(CELERITAS_REAL_TYPE
  "Global runtime precision for real numbers")

if((CELERITAS_CORE_GEO STREQUAL "ORANGE")
    AND (NOT CELERITAS_UNITS STREQUAL "CGS"))
  celeritas_error_incompatible_option(
    "ORANGE currently requires CGS units"
    CELERITAS_UNITS
    CGS
  )
endif()

#----------------------------------------------------------------------------#
# CMAKE VERSION CHECKS
#----------------------------------------------------------------------------#

if(CMAKE_VERSION VERSION_LESS 3.22 AND CELERITAS_USE_HIP)
  message(WARNING "HIP support is immature; CMake 3.22+ is recommended.")
endif()

if(CMAKE_VERSION VERSION_LESS 3.27 AND CELERITAS_USE_covfie)
  # See https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html#id30
  message(FATAL_ERROR "CMake 3.27 or greater is required for covfie support")
endif()

#----------------------------------------------------------------------------#
# CMAKE INTRINSIC OPTIONS
#
# These are generally used to initialize properties on targets, and it's
# possible Celeritas is being built inside another project. If documentation is
# provided and Celeritas is the top-level project, create a cache variable to
# make it easier for the user to override the behavior. Otherwise, create a
# local variable to override the behavior of Celeritas and any projects it
# includes.
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

# Default to using C++17
celeritas_set_default(CMAKE_CXX_STANDARD 17 STRING "C++ standard")
celeritas_set_default(CMAKE_CXX_EXTENSIONS OFF "Allow C++ compiler extensions")
if(CELERITAS_USE_CUDA)
  # Default to setting CUDA C++ standard the same as C++
  celeritas_set_default(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
  celeritas_set_default(CMAKE_CUDA_EXTENSIONS "${CMAKE_CXX_EXTENSIONS}")
  if(NOT CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
    # See https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_HOST_COMPILER.html
    message(WARNING "CUDA host compiler mismatch: "
      "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER} "
      "but CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
  endif()
endif()
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

### Linking flags ###

celeritas_set_default(BUILD_SHARED_LIBS ON
  "Build shared libraries instead of static")
celeritas_set_default(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON
  "Inform installed binaries of external library rpaths")
if(BUILD_SHARED_LIBS)
  # Inform installed binaries of internal library rpaths
  celeritas_set_default(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}")
  # Do not relink libs/binaries when dependent shared libs change
  celeritas_set_default(CMAKE_LINK_DEPENDS_NO_SHARED ON)
endif()
set(_default_pic OFF)
if(BUILD_SHARED_LIBS OR CELERITAS_USE_ROOT)
  set(_default_pic ON)
endif()
celeritas_set_default(CMAKE_POSITION_INDEPENDENT_CODE ${_default_pic}
  "Make sure build can be used with shared libs or runtime modules")
unset(_default_pic)

### Installation flags ###

# When developing add checking for proper usage of `install(`
if(CELERITAS_DEBUG)
  celeritas_set_default(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
endif()
# Avoid printing details about already installed files
celeritas_set_default(CMAKE_INSTALL_MESSAGE LAZY)

#----------------------------------------------------------------------------#
# Output locations for Celeritas products (used by CeleritasUtils.cmake and
# install code below) will mirror the installation layout
set(CELERITAS_CMAKE_CONFIG_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(CELERITAS_HEADER_CONFIG_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}")
set(CELERITAS_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CELERITAS_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

#----------------------------------------------------------------------------#
# DEPENDENCIES
# NOTE: when adding a new dependency, update the CMakeConfig "find_dependency"
# and check the dependency code in the CMakeConfig setup at the bottom of this
# file; with exceptions:
# - CLI11 is only used for app/
# - googletest is only used for test/
#
# TODO: dependencies are ordered from highest to lowest level so that we can
# let other packages choose dependencies. Low-level dependencies should
# be sorted alphabetically.
# LArSoft -> ROOT
# ROOT -> JSON
# G4 -> VecGeom
# VecGeom -> CUDA
# ...
#----------------------------------------------------------------------------#

# Model explicit package dependencies
foreach(_pkg LArSoft DD4hep)
  if(CELERITAS_USE_${_pkg})
    foreach(_dep ROOT Geant4)
      if(NOT CELERITAS_USE_${_dep})
        celeritas_error_incompatible_option(
          "${_pkg} requires ${_dep}" CELERITAS_USE_${_dep} ON
        )
      endif()
    endforeach()
  endif()
endforeach()

#----------------------------------------------------------------------------#

include(CeleritasUtils)

if(CELERITAS_USE_CLI11)
  celeritas_find_or_builtin_package(CLI11 2.4)
endif()

if(BUILD_TESTING)
  # Build the test tree for unit and/or app tests
  include(CTest)
endif()

if(CELERITAS_USE_LArSoft)
  if(NOT LArSoft_FOUND)
    find_package(LArSoft REQUIRED)
  endif()
endif()

if(CELERITAS_USE_DD4hep)
  if(NOT DD4hep_FOUND)
    find_package(DD4hep REQUIRED COMPONENTS DDCore DDG4)
  endif()
endif()

if(CELERITAS_USE_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "The CMAKE_CUDA_ARCHITECTURES variable, necessary "
      "to specify the GPU target device, is not defined, and the default "
      "guess may be suboptimal and cause further issues. "
      "Verify that this value matches your CUDA device capability."
    )
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED QUIET)
elseif(CELERITAS_USE_HIP)
  enable_language(HIP)
  # NOTE: known broken up to 5.6; unknown after
  if(CELERITAS_DEVICE_DEBUG AND "${hip-lang_VERSION}" VERSION_LESS 5.7)
    celeritas_error_incompatible_option(
      "HIP cannot build debug code (unhandled SGPR spill to memory)"
      CELERITAS_DEVICE_DEBUG
      OFF
    )
  endif()
  # Include the default ROCm installation path in the search
  find_library(ROCTX_LIBRARY roctx64 PATHS /opt/rocm/lib)
  celeritas_set_onoff(CELERITAS_HAVE_ROCTX ${ROCTX_LIBRARY})
  find_file(HIPCUB hipcub_version.hpp PATHS /opt/rocm/include/hipcub)
  celeritas_set_onoff(CELERITAS_HAVE_HIPCUB ${HIPCUB})
endif()

if(CELERITAS_USE_CUDA)
  # NOTE: older versions of CUDA/Thrust don't include a CMake configuration
  # file, so don't make it required
  find_package(Thrust)
endif()
# Unconditionally set Thrust availability for downstream configuration
celeritas_force_package(Thrust "${Thrust_FOUND}")

if(CELERITAS_USE_Geant4)
  if(NOT Geant4_FOUND)
    find_package(Geant4 REQUIRED)
  endif()
  if(CELERITAS_OPENMP STREQUAL "track" AND Geant4_multithreaded_FOUND)
    message(WARNING "Track-level OpenMP will conflict with MT runs of geant4:"
      "consider setting CELERITAS_OPENMP to \"event\" or disabling OpenMP"
    )
  endif()
endif()

if(CELERITAS_USE_HepMC3)
  if(NOT HepMC3_FOUND)
    find_package(HepMC3 REQUIRED)
  endif()
  set(HepMC3_LIBRARIES HepMC3::HepMC3)
endif()

celeritas_find_or_builtin_package(nlohmann_json 3.7.0)

if(CELERITAS_USE_MPI)
  find_package(MPI REQUIRED)
endif()

if(CELERITAS_USE_OpenMP)
  if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
  endif()
  if(OpenMP_VERSION AND (OpenMP_VERSION VERSION_LESS "3.0"))
    message(WARNING
      "OpenMP version ${OpenMP_VERSION} may fail to compile"
    )
  endif()
endif()

if(CELERITAS_USE_Perfetto)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    celeritas_error_incompatible_option(
      "Perfetto is not supported with CUDA/HIP build"
      CELERITAS_USE_Perfetto
      OFF
    )
  else()
    celeritas_find_or_builtin_package(Perfetto)
  endif()
endif()

if(CELERITAS_USE_PNG)
  # This will find libpng
  find_package(PNG REQUIRED)
endif()

if(CELERITAS_USE_Python)
  if(NOT Python_FOUND)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)
  endif()
endif()

if(CELERITAS_USE_ROOT)
  if(NOT BUILD_SHARED_LIBS)
    celeritas_error_incompatible_option(
      "ROOT PCM fails at runtime without shared libraries"
      BUILD_SHARED_LIBS
      ON
    )
  endif()
  if(NOT ROOT_FOUND)
    # ROOT requirement is due to missing CMake commands in old versions
    find_package(ROOT 6.24 REQUIRED)
  endif()
  if(ROOT_CXX_STANDARD AND CMAKE_CXX_STANDARD
     AND NOT (ROOT_CXX_STANDARD STREQUAL CMAKE_CXX_STANDARD))
    message(WARNING "ROOT C++ standard (${ROOT_CXX_STANDARD}) "
      "does not match current CMake standard (${CMAKE_CXX_STANDARD}): "
      "build warnings/errors may result")
  endif()
endif()


if(CELERITAS_USE_covfie)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    set(_min_covfie_version 0.14)
  else()
    set(_min_covfie_version 0.13)
  endif()
  celeritas_find_or_builtin_package(covfie ${_min_covfie_version})
endif()

if(CELERITAS_USE_VecGeom)
  if(CELERITAS_USE_CUDA)
    # 1.2.10 is needed due to the `no-as-needed` link feature
    # requiring VecGeom to be build with a distinct object
    # and final libraries
    set(_min_vecgeom_version 1.2.10)
  else()
    # 1.2.8 is needed due to G4VG, which needs the logger.
    set(_min_vecgeom_version 1.2.8)
  endif()
  if(NOT VecGeom_FOUND)
    find_package(VecGeom ${_min_vecgeom_version} REQUIRED)
  elseif(VecGeom_VERSION VERSION_LESS _min_vecgeom_version)
    # Another package, probably Geant4, is already using vecgeom
    celeritas_error_incompatible_option(
      "VecGeom version \"${VecGeom_VERSION}\" at \"${VecGeom_DIR}\" is too old for Celeritas to use: you must update to ${_min_vecgeom_version} or higher"
      CELERITAS_USE_VecGeom
      OFF
    )
  endif()

  # Save VecGeom_NAV variable for diagnostics
  if(VecGeom_VERSION VERSION_LESS 2.0)
    if(VecGeom_USE_NAVINDEX)
      set(VecGeom_NAV "index")
    else()
      set(VecGeom_NAV "path")
    endif()
  endif()

  if(CELERITAS_USE_CUDA)
    if(NOT VecGeom_CUDA_FOUND)
      celeritas_error_incompatible_option(
        "VecGeom installation at \"${VecGeom_DIR}\" is not CUDA-enabled"
        CELERITAS_USE_CUDA
        "${VecGeom_CUDA_FOUND}"
      )
    elseif(VecGeom_NAV STREQUAL "path")
      celeritas_error_incompatible_option(
        "VecGeom installation at \"${VecGeom_DIR}\" was built with VecGeom_NAV=path: use index or tuple instead for CUDA compatibility"
        CELERITAS_USE_VecGeom
        OFF
      )
    elseif(NOT (VecGeom_CUDA_ARCHITECTURES STREQUAL CMAKE_CUDA_ARCHITECTURES))
      message(WARNING "CUDA architecture types between "
        "VecGeom (${VecGeom_CUDA_ARCHITECTURES}) and "
        "Celeritas (${CMAKE_CUDA_ARCHITECTURES}) should probably match: "
        "runtime errors may result"
      )
    endif()
  endif()
  if(CELERITAS_REAL_TYPE STREQUAL "float" AND NOT VecGeom_single_precision_FOUND)
    celeritas_error_incompatible_option(
      "VecGeom installation at \"${VecGeom_DIR}\" uses double precision"
      CELERITAS_REAL_TYPE
      "double"
    )
  endif()
  if(CELERITAS_REAL_TYPE STREQUAL "double" AND VecGeom_single_precision_FOUND)
    celeritas_error_incompatible_option(
      "VecGeom installation at \"${VecGeom_DIR}\" uses single precision"
      CELERITAS_REAL_TYPE
      "float"
    )
  endif()
  if(CELERITAS_BUILD_TESTS AND NOT VecGeom_GDML_FOUND
      AND CELERITAS_CORE_GEO STREQUAL "VecGeom")
    celeritas_error_incompatible_option(
      "VecGeom installation at \"${VecGeom_DIR}\" was not built with VGDML:
      celer-sim and many tests will fail"
      CELERITAS_BUILD_TESTS
      OFF
    )
  endif()
  if(NOT VecGeom_LIBRARIES)
    set(VecGeom_LIBRARIES ${VECGEOM_LIBRARIES})
  endif()
  if(NOT VecGeom_LIBRARIES)
    set(VecGeom_LIBRARIES VecGeom::vecgeom)
  endif()
  if(CELERITAS_USE_Geant4)
    # Enforce minimum version
    celeritas_find_or_builtin_package(G4VG 1.0.3)
  endif()

  # Default to using surface navigation if it's available
  cmake_dependent_option(CELERITAS_VecGeom_SURFACE
    "Use VecGeom surface navigation" "ON"
    "VecGeom_USE_SURF" OFF
  )
endif()

if(CELERITAS_BUILD_DOCS)
  set(_prev_dot "${DOXYGEN_DOT_EXECUTABLE}")
  if(NOT Doxygen_FOUND)
    find_package(Doxygen COMPONENTS doxygen dot)
  endif()
  if(NOT DEFINED Doxygen_doxygen_FOUND)
    # Backward compatibility or caller package doesn't specify components
    set(Doxygen_doxygen_FOUND ${Doxygen_FOUND})
  endif()
  if(NOT Doxygen_doxygen_FOUND)
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
    celeritas_check_python_module(CELERITAS_USE_Breathe "breathe")
    celeritas_check_python_module(CELERITAS_USE_Furo "furo")
    celeritas_check_python_module(CELERITAS_USE_SphinxBibtex "sphinxcontrib.bibtex")
    if(CELERITAS_USE_Breathe)
      find_program(LATEXMK_EXECUTABLE latexmk)
    endif()
  endif()

  if(NOT DOXYGEN_DOT_EXECUTABLE)
    find_program(DOXYGEN_DOT_EXECUTABLE dot)
  endif()
  if(DOXYGEN_DOT_EXECUTABLE AND NOT _prev_dot)
    message(STATUS "Found dot: ${DOXYGEN_DOT_EXECUTABLE}")
  elseif(_prev_dot STREQUAL "")
    # Note: use string comparison, not boolean, to avoid matching old "NOTFOUND"
    message(STATUS "Could NOT find graphviz generator `dot`")
  endif()

  include(ExternalProject)
endif()

if(CELERITAS_USE_GTest)
  celeritas_find_or_builtin_package(GTest 1.10)
endif()

#----------------------------------------------------------------------------#
# System properties

# Availability of sincospi and related functions: all comibinations of:
# `^${CELERITAS_SINCOSPI_PREFIX}(sincos(pi)?|(sin|cos)pi)f?$`
if(NOT DEFINED CELERITAS_SINCOSPI_PREFIX)
  foreach(_prefix "__" "_" "")
    try_compile(CELERITAS_TRY_${_prefix}SINCOSPI
      # NOTE: the line below is required for CMake < 3.25
      "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeScratch/sincospi${_prefix}"
      SOURCES "${PROJECT_SOURCE_DIR}/cmake/try-sincospi.cc"
      COMPILE_DEFINITIONS "-DCELERITAS_SINCOSPI_PREFIX=${_prefix}"
    )
    if(CELERITAS_TRY_${_prefix}SINCOSPI)
      set(CELERITAS_SINCOSPI_PREFIX "${_prefix}" CACHE INTERNAL
        "Prefix for compiler-provided sincospi and related functions"
      )
      break()
    endif()
  endforeach()
endif()

#----------------------------------------------------------------------------#
# BUILTINS
#----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN)
  include(FetchContent)
  add_subdirectory(external)
endif()

#----------------------------------------------------------------------------#
# RDC SUPPORT FOR VECGEOM+CUDA
#----------------------------------------------------------------------------#

# Load our local copy in case a dependency hasn't loaded a newer one
# and install it alongside CeleritasConfig (see further below)
set(_LOCAL_RDCUTILS_FILENAME "${PROJECT_SOURCE_DIR}/cmake/external/CudaRdcUtils.cmake")
include("${_LOCAL_RDCUTILS_FILENAME}")

# NOTE: include library utils after RDC
include(CeleritasLibrary)
include(CeleritasLibraryUtils)

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

# Add the main libraries
add_subdirectory(src)

#----------------------------------------------------------------------------#
# UNIT TESTS
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_TESTS)
  add_subdirectory(test)
endif()

#----------------------------------------------------------------------------#
# APPLICATIONS AND USER EXECUTABLES
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_APPS)
  add_subdirectory(app)
endif()

#----------------------------------------------------------------------------#
# DOCUMENTATION
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DOCS)
  add_subdirectory(doc)
endif()

#----------------------------------------------------------------------------#
# CONFIG FILE INSTALLATION
#----------------------------------------------------------------------------#

# Where to install configured cmake files
set(CELERITAS_INSTALL_CMAKECONFIGDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Build list of CMake files to install
set(_cmake_files
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasLibrary.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasUtils.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/CeleritasG4Tests.cmake"
  # TODO: in the future, don't include or distribute RdcUtils.
  # For now, include in case downstream projects do `include(CudaRdcUtils)`.
  "${_LOCAL_RDCUTILS_FILENAME}"
)
# Add packages with custom Celeritas-provided FindX scripts
foreach(_dep CLI11 covfie Geant4 G4VG HepMC3 LArSoft Perfetto ROOT Thrust VecGeom)
  if(CELERITAS_USE_${_dep} AND NOT CELERITAS_BUILTIN_${_dep})
    list(APPEND _cmake_files "${PROJECT_SOURCE_DIR}/cmake/Find${_dep}.cmake")
  endif()
endforeach()
if(CELERITAS_USE_LArSoft)
  foreach(_dep lardataobj cetmodules)
    list(APPEND _cmake_files "${PROJECT_SOURCE_DIR}/cmake/Find${_dep}.cmake")
  endforeach()
endif()

if(CELERITAS_BUILD_TESTS)
  list(APPEND _cmake_files
    "${PROJECT_SOURCE_DIR}/cmake/CeleritasAddTest.cmake"
  )
endif()

install(FILES ${_cmake_files}
  DESTINATION "${CELERITAS_INSTALL_CMAKECONFIGDIR}"
  COMPONENT development
)

# Copy CMake files to support using Celeritas build dir as an install dir
file(COPY ${_cmake_files}
  DESTINATION "${CELERITAS_INSTALL_CMAKECONFIGDIR}"
)

# Export all cache variables that start with CELERITAS_
set(CELERITAS_EXPORT_VARIABLES)
macro(celeritas_export_var varname)
  # Value of _key may be ;-separated list, so we need to escape the ; if present
  string(REPLACE ";" "\;" _key_val "${${varname}}")
  list(APPEND CELERITAS_EXPORT_VARIABLES "set(${varname} \"${_key_val}\")")
endmacro()
celeritas_export_var(Celeritas_VERSION_STRING)
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Components")
celeritas_export_var(CELERITAS_COMPONENTS)
celeritas_export_var(CELERITAS_ENABLED_COMPONENTS)
celeritas_export_var(CELERITAS_OPTION_VARS)
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Configuration options")
get_directory_property(_cachevar_keys CACHE_VARIABLES)
foreach(_key IN LISTS _cachevar_keys)
  if(_key MATCHES "^CELERITAS_")
    celeritas_export_var(${_key})
  endif()
endforeach()

# Save whether we need GDML requirement
if(Geant4_gdml_FOUND OR VecGeom_GDML_FOUND)
  set(CELERITAS_USE_GDML ON)
else()
  set(CELERITAS_USE_GDML OFF)
endif()
celeritas_export_var(CELERITAS_USE_GDML)
foreach(_key IN LISTS CELERITAS_FORCED_PACKAGE_VARS)
  celeritas_export_var(${_key})
endforeach()

# Export defaulted CMake variables
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Defaulted local variables")
set(_celer_other_cmake_variables)
if(CELERITAS_USE_CUDA)
  set(_celer_other_cmake_variables CMAKE_CUDA_ARCHITECTURES)
elseif(CELERITAS_USE_HIP)
  set(_celer_other_cmake_variables CMAKE_HIP_ARCHITECTURES)
endif()
foreach(_key IN LISTS CELERITAS_DEFAULT_VARIABLES _celer_other_cmake_variables)
  # Value of _key may be ;-separated list, so we need to escape the ; if present
  string(REPLACE ";" "\;" _key_val "${${_key}}")
  list(APPEND CELERITAS_EXPORT_VARIABLES "set(CELERITAS_${_key} \"${_key_val}\")")
endforeach()

# Add hints for direct dependencies and indirect geant dependencies
list(APPEND CELERITAS_EXPORT_VARIABLES "\n# Hints for upstream dependencies")
foreach(_key
  MPIEXEC_EXECUTABLE CUDAToolkit_BIN_DIR
  covfie_DIR Geant4_DIR GTest_DIR HepMC3_DIR nlohmann_json_DIR
  Python_DIR ROOT_DIR VecCore_DIR VecGeom_DIR G4VG_DIR
  CLHEP_DIR ZLIB_DIR EXPAT_DIR XercesC_DIR PTL_DIR
  EXPAT_INCLUDE_DIR EXPAT_LIBRARY
  VDT_INCLUDE_DIR VDT_LIBRARY
  XercesC_LIBRARY XercesC_INCLUDE_DIR
)
  set(_val "${${_key}}")
  if(_val)
    set(_cache_val "$CACHE{${_key}}")
    if(_cache_val)
      list(APPEND CELERITAS_EXPORT_VARIABLES
        "_celer_set_cache_default(${_key} \"${_cache_val}\" PATH \"Set by CeleritasConfig.cmake\")"
      )
    else()
    list(APPEND CELERITAS_EXPORT_VARIABLES
      "_celer_set_local_default(${_key} \"${_val}\")"
    )
  endif()
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

# Export targets to the build tree
export(EXPORT celeritas-targets
  FILE "${CELERITAS_CMAKE_CONFIG_DIRECTORY}/CeleritasTargets.cmake"
  NAMESPACE Celeritas::
)

if(Celeritas_VERSION VERSION_EQUAL "0.0.0")
  install(CODE "
message(WARNING \"The Celeritas version was not detected during configuration.
  (Check the beginning of your initial configure output for more details.)
  This will result in the installation having incorrect version metadata and
  will interfere with downstream CMake version requirements and may obscure
  provenance data in output results.\")
message(WARNING \"If using a git checkout, please run
  `git fetch --all --tags`, rebuild with `cmake --build ${CMAKE_BINARY_DIR}`,
  and install with `cmake --install ${CMAKE_BINARY_DIR}`.\")
")
endif()
if(CELERITAS_USE_HIP AND NOT BUILD_SHARED_LIBS)
  # See celeritas_device_toolkit above
  install(CODE "
message(WARNING \"CMake may not be able to correctly propagate implicit HIP
  libraries: downstream executables may fail to link.\")
")
endif()

#----------------------------------------------------------------------------#
