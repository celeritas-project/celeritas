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
celeritas_add_test(base/ArrayUtils.test.cc)
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
# Physics

celeritas_setup_tests(SERIAL PREFIX physics/base)

add_cudaoptional_test(physics/base/Particle)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_add_test(random/distributions/BernoulliDistribution.test.cc)
celeritas_add_test(random/distributions/IsotropicDistribution.test.cc)
celeritas_add_test(random/distributions/RadialDistribution.test.cc)
celeritas_add_test(random/distributions/UniformRealDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(
    random/distributions/IsotropicDistribution.test.cu SUFFIX cuda
  )
  celeritas_add_test(
    random/distributions/RadialDistribution.test.cu SUFFIX cuda
  )
  celeritas_add_test(
    random/distributions/UniformRealDistribution.test.cu SUFFIX cuda
  )
  celeritas_add_test(random/cuda/RngEngine.test.cu SUFFIX cuda)
endif()

#-----------------------------------------------------------------------------#
