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

function(add_cudaoptional_test base)
  if(CELERITAS_USE_CUDA)
    set(_sources SOURCES "${base}.test.cu")
  endif()
  celeritas_add_test("${base}.test.cc" ${_sources} ${ARGN})
endfunction()

if(NOT CELERITAS_USE_CUDA)
  set(_needs_cuda DISABLE)
endif()

#-----------------------------------------------------------------------------#
# Gtest additions
celeritas_setup_tests(SERIAL PREFIX gtest/detail)

celeritas_add_test(gtest/detail/test/Macros.test.cc)

#-----------------------------------------------------------------------------#
# Base

celeritas_setup_tests(SERIAL PREFIX base)

celeritas_add_test(base/Array.test.cc)
celeritas_add_test(base/DeviceAllocation.test.cc)
celeritas_add_test(base/DeviceVector.test.cc)
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

  celeritas_add_test(geometry/GeoTrackView.test.cc)

  # CUDA tests: link cuda sources as needed
  if(CELERITAS_USE_CUDA)
    # See https://github.com/celeritas-project/celeritas/pull/10 for a
    # discussion of the failures from trying to directly build an executable
    # from code with CUDA separable compilation. This extra test library is
    # necessary.
    add_library(celeritas_vgtest geometry/GeoTrackView.test.cu)
    target_link_libraries(celeritas_vgtest
      PRIVATE
      celeritas
      VecGeom::vecgeomcuda
      VecGeom::vecgeomcuda_static
    )
    set_target_properties(celeritas_vgtest PROPERTIES
      LINKER_LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(geometry_GeoTrackView celeritas_vgtest)
  endif()
endif()

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)
celeritas_add_test(random/RadialDistribution.test.cc)
celeritas_add_test(random/UniformRealDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(random/RadialDistribution.test.cu SUFFIX cuda)
  celeritas_add_test(random/UniformRealDistribution.test.cu SUFFIX cuda)
  celeritas_add_test(random/RngEngine.test.cu SUFFIX cuda)
endif()

#-----------------------------------------------------------------------------#
