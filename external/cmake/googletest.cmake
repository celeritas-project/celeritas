#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

googletest
----------

Include this file *directly* from the parent directory to set up GoogleTest.

#]=======================================================================]

# Set up options
set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
set(gtest_disable_pthreads OFF CACHE BOOL "")
if(MSVC)
  # Downstream tests get compiled with `/MDd` for some reason, and if this is
  # 'off' they'll instead be compiled with '-MTd', causing nasty link errors
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(googletest)

# Define the same library alias used when Googletest is installed externally
add_library(gtest_celeritas INTERFACE)
target_link_libraries(gtest_celeritas INTERFACE gtest)
add_library(GTest::GTest ALIAS gtest_celeritas)

# Google compile definitions don't get propagated correctly
target_compile_definitions(gtest_celeritas INTERFACE
  $<BUILD_INTERFACE:GTEST_LINKED_AS_SHARED_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>>
  $<BUILD_INTERFACE:GTEST_HAS_PTHREAD=$<BOOL:${gtest_disable_pthreads}>>
)

if(MSVC)
  # Hide "... static storage duration was declared but never referenced"
  target_compile_options(gtest_celeritas INTERFACE
    $<BUILD_INTERFACE:/Qdiag-disable:2415>)
endif()

if(INSTALL_GTEST)
  install(TARGETS gtest_celeritas
    EXPORT GTest
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    COMPONENT GTest
  )
endif()

#-----------------------------------------------------------------------------#
