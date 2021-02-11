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

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
  target_link_libraries(CeleritasTest
    PRIVATE
      VecGeom::vecgeomcuda
      VecGeom::vecgeomcuda_static
  )
endif()

target_include_directories(CeleritasTest
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

function(celeritas_cudaoptional_test base)
  if(CELERITAS_USE_CUDA)
    set(_cuda_args GPU SOURCES "${base}.test.cu")
  endif()
  celeritas_add_test("${base}.test.cc" ${_cuda_args} ${ARGN})
endfunction()

set(_not_impl DISABLE)

#-----------------------------------------------------------------------------#
# Gtest additions
celeritas_setup_tests(SERIAL PREFIX gtest/detail)

celeritas_add_test(gtest/detail/test/Macros.test.cc)

#-----------------------------------------------------------------------------#
# Base

celeritas_setup_tests(SERIAL PREFIX base)

celeritas_add_test(base/Algorithms.test.cc)
celeritas_add_test(base/Array.test.cc)
celeritas_add_test(base/ArrayUtils.test.cc)
celeritas_add_test(base/Constants.test.cc)
celeritas_add_test(base/DeviceAllocation.test.cc GPU)
celeritas_add_test(base/DeviceVector.test.cc GPU)
celeritas_add_test(base/Interpolator.test.cc)
celeritas_add_test(base/Join.test.cc)
celeritas_add_test(base/OpaqueId.test.cc)
celeritas_add_test(base/Quantity.test.cc)
celeritas_add_test(base/SoftEqual.test.cc)
celeritas_add_test(base/Span.test.cc)
celeritas_add_test(base/SpanRemapper.test.cc)
celeritas_add_test(base/Stopwatch.test.cc)
celeritas_add_test(base/TypeDemangler.test.cc)
celeritas_add_test(base/VectorUtils.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(base/NumericLimits.test.cc
    SOURCES base/NumericLimits.test.cu
    GPU)
endif()

celeritas_cudaoptional_test(base/Pie)
celeritas_cudaoptional_test(base/Range)
celeritas_cudaoptional_test(base/StackAllocator)

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)

celeritas_add_test(comm/Communicator.test.cc)
celeritas_add_test(comm/Logger.test.cc)

#-----------------------------------------------------------------------------#
# Geometry

if(CELERITAS_USE_VecGeom)
  celeritas_setup_tests(SERIAL PREFIX geometry
    LINK_LIBRARIES VecGeom::vecgeom)

  # CUDA tests: link cuda sources as needed
  if(CELERITAS_USE_CUDA)
    # See https://github.com/celeritas-project/celeritas/pull/10 for a
    # discussion of the failures from trying to directly build an executable
    # from code with CUDA separable compilation. This extra test library is
    # necessary.
    add_library(celeritas_vgtest
      geometry/GeoTrackView.test.cu
      geometry/LinearPropagator.test.cu
    )
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
    list(APPEND CELERITASTEST_LINK_LIBRARIES celeritas_vgtest)
  endif()

  celeritas_add_test(geometry/GeoParams.test.cc GPU)
  celeritas_add_test(geometry/GeoTrackView.test.cc GPU)
  celeritas_add_test(geometry/LinearPropagator.test.cc GPU)
endif()

#-----------------------------------------------------------------------------#
# I/O (ROOT)

celeritas_setup_tests(SERIAL PREFIX io)

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()
if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()

celeritas_add_test(io/RootImporter.test.cc ${_needs_root}
  LINK_LIBRARIES Celeritas::ROOT)
celeritas_add_test(io/EventReader.test.cc ${_needs_hepmc})

#-----------------------------------------------------------------------------#
# Physics

add_library(CeleritasPhysicsTest
  physics/InteractionIO.cc
  physics/InteractorHostTestBase.cc
  physics/SecondaryIO.cc
  physics/base/MockModel.cc
  physics/base/MockProcess.cc
  physics/base/PhysicsTestBase.cc
)
target_link_libraries(CeleritasPhysicsTest PRIVATE celeritas CeleritasTest)
set(CELERITASTEST_LINK_LIBRARIES CeleritasPhysicsTest)

celeritas_setup_tests(SERIAL PREFIX physics/base)
celeritas_cudaoptional_test(physics/base/Particle)
celeritas_add_test(physics/base/Physics.test.cc)
celeritas_add_test(physics/base/PhysicsStepUtils.test.cc ${_not_impl})

celeritas_setup_tests(SERIAL PREFIX physics/grid)
celeritas_add_test(physics/grid/GridIdFinder.test.cc)
celeritas_add_test(physics/grid/PhysicsGridCalculator.test.cc)
celeritas_add_test(physics/grid/UniformGrid.test.cc)
celeritas_add_test(physics/grid/ValueGridBuilder.test.cc)
celeritas_add_test(physics/grid/ValueGridInserter.test.cc)

celeritas_setup_tests(SERIAL PREFIX physics/material)
celeritas_add_test(physics/material/ElementSelector.test.cc)
celeritas_cudaoptional_test(physics/material/Material)

#-----------------------------------------------------------------------------#
# Physics (EM)

celeritas_setup_tests(SERIAL PREFIX physics/em)

# NOTE: Remove '${_not_impl}' below at the start of developing each class.
celeritas_add_test(physics/em/BetheBloch.test.cc ${_not_impl})
celeritas_add_test(physics/em/BetheHeitler.test.cc)
celeritas_add_test(physics/em/BremRel.test.cc ${_not_impl})
celeritas_add_test(physics/em/EPlusGG.test.cc)
celeritas_add_test(physics/em/KleinNishina.test.cc)
celeritas_add_test(physics/em/LivermorePE.test.cc)
celeritas_add_test(physics/em/MollerBhabha.test.cc)
celeritas_add_test(physics/em/Rayleigh.test.cc ${_not_impl})
celeritas_add_test(physics/em/Urban.test.cc ${_not_impl})
celeritas_add_test(physics/em/Wentzel.test.cc ${_not_impl})

# END PHYSICS TESTS
set(CELERITASTEST_LINK_LIBRARIES)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_add_test(random/distributions/BernoulliDistribution.test.cc)
celeritas_add_test(random/distributions/ExponentialDistribution.test.cc)
celeritas_add_test(random/distributions/IsotropicDistribution.test.cc)
celeritas_add_test(random/distributions/RadialDistribution.test.cc)
celeritas_add_test(random/distributions/UniformRealDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(random/cuda/RngEngine.test.cu GPU)
endif()

#-----------------------------------------------------------------------------#
# Sim

celeritas_setup_tests(SERIAL PREFIX sim)
if(CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  celeritas_add_test(sim/TrackInitializerStore.test.cc GPU
    SOURCES sim/TrackInitializerStore.test.cu
    LINK_LIBRARIES VecGeom::vecgeom)
endif()

#-----------------------------------------------------------------------------#
