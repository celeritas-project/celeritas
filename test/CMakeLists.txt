#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# TEST LIBRARY
#-----------------------------------------------------------------------------#

add_library(CeleritasTest
  gtest/Test.cc
  gtest/detail/NonMasterResultPrinter.cc
  gtest/detail/ParallelHandler.cc
  gtest/detail/TestMain.cc
  gtest/detail/Utils.cc
)
add_library(Celeritas::Test ALIAS CeleritasTest)
target_link_libraries(CeleritasTest PUBLIC celeritas GTest::GTest)

target_include_directories(CeleritasTest
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#
# Base

celeritas_setup_tests(SERIAL PREFIX base)
celeritas_add_test(base/Array.test.cc)
celeritas_add_test(base/Interpolator.test.cc)
celeritas_add_test(base/OpaqueId.test.cc)
celeritas_add_test(base/Range.test.cc)
celeritas_add_test(base/SoftEqual.test.cc)
celeritas_add_test(base/Span.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(base/NumericLimits.test.cc
    SOURCES base/NumericLimits.test.cu)
  celeritas_add_test(base/StackAllocator.test.cc
    SOURCES base/StackAllocator.test.cu)
endif()

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)
celeritas_add_test(comm/Communicator.test.cc)

#-----------------------------------------------------------------------------#
# Geometry

if(CELERITAS_USE_VECGEOM)
  celeritas_setup_tests(SERIAL PREFIX geometry
    LINK_LIBRARIES VecGeom::vecgeom)

  celeritas_add_test(geometry/VGHost.test.cc)

  # CUDA tests: link cuda sources as needed
  if(CELERITAS_USE_CUDA)
    list(APPEND CELERITASTEST_LINK_LIBRARIES
      VecGeom::vecgeomcuda
      VecGeom::vecgeomcuda_static)

    celeritas_add_test(geometry/VGGeometry.test.cc
      SOURCES geometry/VGGeometry.test.cu)
    # XXX use ADDED_TARGET instead of manually entering target name
    set_target_properties(geometry_VGGeometry PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
    )
  else()
    celeritas_add_test(geometry/VGGeometry.test.cc)
  endif()

  set(CELERITASTEST_LINK_LIBRARIES)
endif()

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)
celeritas_add_test(random/RadialDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(random/RadialDistribution.test.cu SUFFIX cuda)
endif()

#-----------------------------------------------------------------------------#
