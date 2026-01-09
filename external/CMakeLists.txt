#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# Download the package, save version, add to a list
set(_celer_builtin_packages)
macro(celer_fetch_external name version url sha)
  if(CELERITAS_BUILTIN_${name} AND NOT CELERITAS_USE_${name})
    message(WARNING "Ignoring CELERITAS_BUILTIN_${name} "
      "since CELERITAS_USE_${name} is disabled")
  else()
    # Save version and annotated string for corecel/Config.cc include
    # Note that since this a macro, PARENT_SCOPE will be the top-level
    # celeritas/CMakeLists.txt
    set(${name}_VERSION ${version} PARENT_SCOPE)
    set(${name}_VERSION_STRING "${version}+builtin" PARENT_SCOPE)

    string(REGEX REPLACE "@VERSION@" "${version}" _url "${url}")
    FetchContent_Declare(
      "${name}"
      URL "${_url}"
      URL_HASH SHA256=${sha}
      DOWNLOAD_NO_PROGRESS TRUE
    )
    list(APPEND _celer_builtin_packages "${name}")
  endif()
endmacro()

#-----------------------------------------------------------------------------#

# Print extra information about downloading: hash failures, for example, are
# silent otherwise
set(CMAKE_MESSAGE_LOG_LEVEL VERBOSE)

#-----------------------------------------------------------------------------#
# CLI11
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_CLI11)
  celer_fetch_external(
    CLI11 2.5.0
    "https://github.com/CLIUtils/CLI11/archive/refs/tags/v@VERSION@.tar.gz"
    17e02b4cddc2fa348e5dbdbb582c59a3486fa2b2433e70a0c3bacb871334fd55
  )

  set(CLI11_BUILD_EXAMPLES OFF)
  set(CLI11_BUILD_DOCS OFF)
  set(CLI11_BUILD_TESTS OFF)
  set(CLI11_PRECOMPILED ON)
endif()

#-----------------------------------------------------------------------------#
# Covfie
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_covfie)
  celer_fetch_external(
    covfie 0.15.3
    "https://github.com/acts-project/covfie/archive/refs/tags/v@VERSION@.tar.gz"
    72da1147c44731caf9163f3931de78d7605a44f056f22a2f6ea024ad02a1ba71
  )

  set(COVFIE_PLATFORM_CPU ON)
  set(COVFIE_PLATFORM_CUDA ${CELERITAS_USE_CUDA})
  set(COVFIE_PLATFORM_HIP ${CELERITAS_USE_HIP})
  set(COVFIE_PLATFORM_SYCL OFF)
  set(COVFIE_BUILD_BENCHMARKS OFF)
  set(COVFIE_BUILD_EXAMPLES OFF)
  set(COVFIE_BUILD_TESTS OFF)
  set(COVFIE_FAIL_ON_WARNINGS OFF)
endif()

#-----------------------------------------------------------------------------#
# GTest
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_GTest)
  celer_fetch_external(
    GTest 1.17.0
    "https://github.com/google/googletest/archive/refs/tags/v@VERSION@.tar.gz"
    65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  )

  set(BUILD_GMOCK OFF) # For compatibility
  set(INSTALL_GTEST OFF)
  set(gtest_hide_internal_symbols ON)
  set(gtest_force_shared_crt ON) # Don't replace /MD with -MT for windows
endif()

#-----------------------------------------------------------------------------#
# nlohmann_json
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_nlohmann_json)
  celer_fetch_external(
    nlohmann_json 3.12.0
    "https://github.com/nlohmann/json/releases/download/v@VERSION@/json.tar.xz"
    42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  )

  set(JSON_Install ON)
endif()

#-----------------------------------------------------------------------------#
# Perfetto
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_Perfetto)
  if(PERFETTO_ROOT AND NOT FETCHCONTENT_SOURCE_DIR_PERFETTO)
    set(FETCHCONTENT_SOURCE_DIR_PERFETTO "${PERFETTO_ROOT}")
  endif()
  celer_fetch_external(
    Perfetto 49.0
    "https://github.com/google/perfetto/archive/refs/tags/v@VERSION@.tar.gz"
    0871a92a162ac5655b7d724f9359b15a75c4e92472716edbc4227a64a4680be4
  )
endif()

#-----------------------------------------------------------------------------#
# G4VG
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_G4VG)
  celer_fetch_external(
    G4VG 1.0.5
    "https://github.com/celeritas-project/g4vg/releases/download/v@VERSION@/g4vg-@VERSION@.tar.gz"
    fa44d071aaf95de596f7b22d2062830acd7e154a3bc9d67ad5a306eff64630d2
  )
  foreach(_var BUILD_TESTS DEBUG)
    set(G4VG_${_var} ${CELERITAS_${_var}})
  endforeach()
endif()

#-----------------------------------------------------------------------------#
# BUILD
#-----------------------------------------------------------------------------#

# Save some variables to see if badly behaving packages change them
set(_checked_cmake_variables
  BUILD_SHARED_LIBS CMAKE_CXX_STANDARD CMAKE_CXX_EXTENSIONS
)
if(NOT WIN32)
  # RPath doesn't matter on windows, and some CI externals change them
  list(APPEND _checked_cmake_variables
    CMAKE_INSTALL_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
  )
endif()
foreach(_var IN LISTS _checked_cmake_variables)
  set(_saved_${_var} "${${_var}}")
endforeach()

# Activate externals
FetchContent_MakeAvailable(${_celer_builtin_packages})

# Check that variables are unmodified
foreach(_var IN LISTS _checked_cmake_variables)
  set(_orig "${_saved_${_var}}")
  set(_new "${${_var}}")
  if(NOT "${_new}" STREQUAL "${_orig}")
    message(SEND_ERROR "An externally fetched package changed the value of ${_var} from ${_orig} to ${_new}. We are saving the original value as a cache variable: please reconfigure.")
    set(${_var} "${_orig}" CACHE STRING "Reset to Celeritas default" FORCE)
  endif()
endforeach()

#-----------------------------------------------------------------------------#
# POST-FETCH CONFIG
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_Perfetto AND perfetto_POPULATED)
  set(Perfetto_INCLUDE_DIR "${perfetto_SOURCE_DIR}/sdk")
  add_library(perfetto STATIC "${Perfetto_INCLUDE_DIR}/perfetto.cc")
  target_include_directories(perfetto
    INTERFACE "$<BUILD_INTERFACE:${Perfetto_INCLUDE_DIR}>"
  )
  add_library(Celeritas::perfetto ALIAS perfetto)
  install(TARGETS perfetto
    EXPORT celeritas-targets
    COMPONENT runtime
  )
endif()

if(CELERITAS_BUILTIN_CLI11)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # __cplusplus: use standard-compliant C++ version macro
    # preprocessor: use standard-compliant preprocessor
    # EHsc: enable standard C++ exception handling
    target_compile_options(CLI11 PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor /EHsc>
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
