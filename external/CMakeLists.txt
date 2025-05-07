#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# Print extra information about downloading: hash failures, for example, are
# silent otherwise
set(CMAKE_MESSAGE_LOG_LEVEL VERBOSE)

# Download the package and add to a list
set(_celer_builtin_packages)
macro(celer_fetch_external name url sha)
  FetchContent_Declare(
    "${name}"
    URL "${url}"
    URL_HASH SHA256=${sha}
    DOWNLOAD_NO_PROGRESS TRUE
  )
  list(APPEND _celer_builtin_packages "${name}")
endmacro()

#-----------------------------------------------------------------------------#
# CLI11
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_CLI11)
  celer_fetch_external(
    CLI11
    "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.5.0.tar.gz"
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
    covfie
    "https://github.com/acts-project/covfie/archive/refs/tags/v0.14.0.tar.gz"
    b4d8afa712c6fc0e2bc6474367d65fad652864b18d0255c5f2c18fd4c6943993
  )
  set(COVFIE_PLATFORM_CPU ON)
  set(COVFIE_PLATFORM_CUDA ${CELERITAS_USE_CUDA})
  set(COVFIE_PLATFORM_HIP ${CELERITAS_USE_HIP})
endif()

#-----------------------------------------------------------------------------#
# GTest
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_GTest)
  celer_fetch_external(
    GTest
    "https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz"
    7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
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
    nlohmann_json
    "https://github.com/nlohmann/json/archive/v3.11.3.tar.gz"
    0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
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
    Perfetto
    "https://github.com/google/perfetto/archive/refs/tags/v49.0.tar.gz"
    0871a92a162ac5655b7d724f9359b15a75c4e92472716edbc4227a64a4680be4
  )
endif()

#-----------------------------------------------------------------------------#
# G4VG
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILTIN_G4VG)
  celer_fetch_external(
    G4VG
    "https://github.com/celeritas-project/g4vg/releases/download/v1.0.4/g4vg-1.0.4.tar.gz"
    ac38878aedaf9fe628f498290cf4fa6cdd21980b7b83663ce8ef3c8f3ff31cb2
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
