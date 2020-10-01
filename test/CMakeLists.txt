#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# TEST LIBRARY
#-----------------------------------------------------------------------------#

set(CELERITAS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
configure_file(gtest/detail/TestConfig.cc.in TestConfig.cc @ONLY)

set(CELER_SOURCES)
list(APPEND CELER_SOURCES
  gtest/Test.cc
  gtest/detail/NonMasterResultPrinter.cc
  gtest/detail/ParallelHandler.cc
  gtest/detail/TestMain.cc
  gtest/detail/Utils.cc
  "${CMAKE_CURRENT_BINARY_DIR}/TestConfig.cc"
)

if(CELERITAS_USE_VecGeom)
  list(APPEND CELER_SOURCES
    geometry/GeoParamsTest.cc
  )
endif()

add_library(CeleritasTest ${CELER_SOURCES})
add_library(Celeritas::Test ALIAS CeleritasTest)
target_link_libraries(CeleritasTest PUBLIC celeritas GTest::GTest)

if(CELERITAS_USE_VecGeom)
  if(CELERITAS_USE_CUDA)
    target_link_libraries(CeleritasTest
       PRIVATE
          VecGeom::vecgeomcuda
          VecGeom::vecgeomcuda_static
    )
  endif(CELERITAS_USE_CUDA)
endif(CELERITAS_USE_VecGeom)

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
celeritas_add_test(base/Quantity.test.cc)
celeritas_add_test(base/Range.test.cc)
celeritas_add_test(base/SoftEqual.test.cc)
celeritas_add_test(base/Span.test.cc)
celeritas_add_test(base/Stopwatch.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(base/NumericLimits.test.cc
    SOURCES base/NumericLimits.test.cu)
endif()

add_cudaoptional_test(base/StackAllocator)

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)

celeritas_add_test(comm/Communicator.test.cc)

#-----------------------------------------------------------------------------#
# Geometry

if(CELERITAS_USE_VecGeom)
  celeritas_setup_tests(SERIAL PREFIX geometry
    LINK_LIBRARIES VecGeom::vecgeom)

  celeritas_add_test(geometry/GeoParams.test.cc)
  celeritas_add_test(geometry/GeoTrackView.test.cc)
  celeritas_add_test(geometry/LinearPropagator.test.cc)

  # CUDA tests: link cuda sources as needed
  if(CELERITAS_USE_CUDA)
    # See https://github.com/celeritas-project/celeritas/pull/10 for a
    # discussion of the failures from trying to directly build an executable
    # from code with CUDA separable compilation. This extra test library is
    # necessary.
    add_library(celeritas_vgtest geometry/GeoTrackView.test.cu geometry/LinearPropagator.test.cu)
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
    target_link_libraries(geometry_GeoParams celeritas_vgtest)
    target_link_libraries(geometry_GeoTrackView celeritas_vgtest)
    target_link_libraries(geometry_LinearPropagator celeritas_vgtest)
  endif()
endif()

#-----------------------------------------------------------------------------#
# I/O (ROOT)

celeritas_setup_tests(SERIAL PREFIX io LINK_LIBRARIES Celeritas::IO)

if(CELERITAS_USE_ROOT)
  celeritas_add_test(io/GeantImporter.test.cc)
endif()
if(CELERITAS_USE_HepMC3)
  celeritas_add_test(io/EventReader.test.cc)
  target_link_libraries(io_EventReader ${HEPMC3_LIB})
endif()

#-----------------------------------------------------------------------------#
# Physics

add_library(CeleritasPhysicsTest
  physics/InteractionIO.cc
  physics/InteractorHostTestBase.cc
  physics/SecondaryIO.cc
)
target_link_libraries(CeleritasPhysicsTest PRIVATE celeritas CeleritasTest)
set(CELERITASTEST_LINK_LIBRARIES CeleritasPhysicsTest)

celeritas_setup_tests(SERIAL PREFIX physics/base)

add_cudaoptional_test(physics/base/Particle)

#-----------------------------------------------------------------------------#
# Physics (EM)

celeritas_setup_tests(SERIAL PREFIX physics/em)

celeritas_add_test(physics/em/KleinNishinaInteractor.test.cc)

# END PHYSICS TESTS
set(CELERITASTEST_LINK_LIBRARIES)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_add_test(random/distributions/BernoulliDistribution.test.cc)
celeritas_add_test(random/distributions/IsotropicDistribution.test.cc)
celeritas_add_test(random/distributions/RadialDistribution.test.cc)
celeritas_add_test(random/distributions/UniformRealDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(random/cuda/RngEngine.test.cu)
endif()

#-----------------------------------------------------------------------------#
