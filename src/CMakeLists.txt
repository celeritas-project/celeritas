#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

# Add a library for propagating common build flags and Celeritas includes
celeritas_add_src_library(BuildFlags INTERFACE)

# Require at least C++17
target_compile_features(BuildFlags INTERFACE cxx_std_17)
if(CELERITAS_USE_CUDA)
  target_compile_features(BuildFlags INTERFACE cuda_std_17)
endif()
if(WIN32)
  target_compile_definitions(BuildFlags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:NOMINMAX NOGDI>
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # __cplusplus: use standard-compliant C++ version macro
    # preprocessor: use standard-compliant preprocessor
    # EHsc: enable standard C++ exception handling
    target_compile_options(BuildFlags INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor /EHsc>
    )
  endif()

  # Inheriting via dominance is correct behavior
  target_compile_options(BuildFlags
    INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/wd4267$<SEMICOLON>/wd4250>"
  )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  # GCC 13 and 14 have *lots* of false positives, and not just on our project
  # see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107532
  celeritas_target_compile_options(BuildFlags
    INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>"
  )
endif()

celeritas_target_include_directories(BuildFlags
  INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CELERITAS_HEADER_CONFIG_DIRECTORY}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

#----------------------------------------------------------------------------#

# Add individual libraries
add_subdirectory(corecel)
add_subdirectory(geocel)
add_subdirectory(orange)
add_subdirectory(celeritas)
if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
endif()

# Install header files
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
  FILES_MATCHING REGEX ".*\\.hh?$"
)

#---------------------------------------------------------------------------##
