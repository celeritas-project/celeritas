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

add_library(CeleritasTest ${CELER_SOURCES})
add_library(Celeritas::Test ALIAS CeleritasTest)
target_link_libraries(CeleritasTest PRIVATE celeritas_cuda PUBLIC GTest::GTest)

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

if(NOT CELERITAS_USE_CUDA)
  set(_needs_cuda DISABLE)
endif()
if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()
if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
else()
  # Optional dependence on low-energy EM data
  set(_ds G4EMLOW)
  set(_optional_geant4
     ENVIRONMENT "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}"
  )
  set(_needs_geant4 ${_optional_geant4})
endif()

#-----------------------------------------------------------------------------#
# Gtest additions
celeritas_setup_tests(SERIAL PREFIX gtest/detail)

celeritas_add_test(gtest/detail/test/Macros.test.cc LINK_LIBRARIES celeritas_final)

#-----------------------------------------------------------------------------#
# Base

celeritas_setup_tests(SERIAL PREFIX base)

celeritas_add_test(base/Algorithms.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Array.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/ArrayUtils.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Constants.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/DeviceAllocation.test.cc GPU LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/DeviceVector.test.cc GPU LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Join.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/OpaqueId.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Quantity.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/ScopedStreamRedirect.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/SoftEqual.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Span.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/SpanRemapper.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/Stopwatch.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/TypeDemangler.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(base/VectorUtils.test.cc LINK_LIBRARIES celeritas_final)

celeritas_cudaoptional_test(base/Collection LINK_LIBRARIES celeritas_final)
celeritas_cudaoptional_test(base/NumericLimits LINK_LIBRARIES celeritas_final)
celeritas_cudaoptional_test(base/Range LINK_LIBRARIES celeritas_final)
celeritas_cudaoptional_test(base/StackAllocator LINK_LIBRARIES celeritas_final)

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)

celeritas_add_test(comm/Communicator.test.cc LINK_LIBRARIES celeritas_final )
celeritas_add_test(comm/Logger.test.cc LINK_LIBRARIES celeritas_final)


#-----------------------------------------------------------------------------#
# Field

celeritas_setup_tests(SERIAL PREFIX field)

celeritas_cudaoptional_test(field/RungeKutta LINK_LIBRARIES celeritas_final)
celeritas_cudaoptional_test(field/FieldDriver LINK_LIBRARIES celeritas_final)

if(CELERITAS_USE_VecGeom)
  if(CELERITAS_USE_CUDA)
    celeritas_add_library(CeleritasPropagatorTest field/FieldPropagator.test.cu)
    target_link_libraries(CeleritasPropagatorTest_final
      PUBLIC celeritas
    )
    target_link_libraries(CeleritasPropagatorTest_objects
      PRIVATE celeritas
    )
    list(APPEND CELERITASTEST_LINK_LIBRARIES CeleritasPropagatorTest)
    celeritas_link_vecgeom_cuda(CeleritasPropagatorTest)
    set(_CeleritasPropagatorTest CeleritasPropagatorTest)
  endif()

  celeritas_add_test(field/FieldPropagator.test.cc GPU
    LINK_LIBRARIES VecGeom::vecgeom ${_CeleritasPropagatorTest})
endif()

#-----------------------------------------------------------------------------#
# Geometry

if(CELERITAS_USE_VecGeom)
  # See https://github.com/celeritas-project/celeritas/pull/10 for a
  # discussion of the failures from trying to directly build an executable
  # from code with CUDA separable compilation. This extra test library is
  # necessary. Furthermore, celeritas must be linked *before* vecgeom. Using
  # PUBLIC library linkage with Celeritas causes headaches in this respect --
  # that's why CeleritasTest and this library use PRIVATE.
  set(CELER_VGTEST_SOURCES
    geometry/GeoTestBase.cc
  )
  if(CELERITAS_USE_CUDA)
    list(APPEND CELER_VGTEST_SOURCES
      geometry/Geo.test.cu
      geometry/LinearPropagator.test.cu
    )
  endif()
  celeritas_add_library(CeleritasVecgeomTest ${CELER_VGTEST_SOURCES})
  target_link_libraries(CeleritasVecgeomTest_final
     PRIVATE Celeritas::Test celeritas_cuda VecGeom::vecgeom
     PUBLIC celeritas_cuda
  )

  target_link_options(CeleritasVecgeomTest_final
     PRIVATE $<DEVICE_LINK:$<TARGET_FILE:celeritas_static>>
  )
  #print_target_properties(CeleritasVecgeomTest_final)
  #print_target_properties(celeritas_final)
  target_link_libraries(CeleritasVecgeomTest_objects
    PRIVATE Celeritas::Test celeritas_cuda VecGeom::vecgeom
  )
  if(CELERITAS_USE_CUDA)
    celeritas_link_vecgeom_cuda(CeleritasVecgeomTest_final)
  endif()

  celeritas_setup_tests(SERIAL PREFIX geometry
    LINK_LIBRARIES CeleritasVecgeomTest VecGeom::vecgeom)

  celeritas_add_test(geometry/GeoMaterial.test.cc ${_needs_root}
    LINK_LIBRARIES Celeritas::ROOT)
  celeritas_add_test(geometry/Geo.test.cc GPU)
  celeritas_add_test(geometry/LinearPropagator.test.cc GPU)
endif()

#-----------------------------------------------------------------------------#
# I/O (ROOT)

celeritas_setup_tests(SERIAL PREFIX io)

celeritas_add_test(io/RootImporter.test.cc ${_needs_root}
  LINK_LIBRARIES celeritas_final Celeritas::ROOT)
celeritas_add_test(io/EventReader.test.cc ${_needs_hepmc} LINK_LIBRARIES celeritas_final)
celeritas_add_test(io/SeltzerBergerReader.test.cc ${_needs_geant4} LINK_LIBRARIES celeritas_final)

#-----------------------------------------------------------------------------#
# Physics

add_library(CeleritasPhysicsTest
  physics/InteractionIO.cc
  physics/InteractorHostTestBase.cc
  physics/SecondaryIO.cc
  physics/base/MockModel.cc
  physics/base/MockProcess.cc
  physics/base/PhysicsTestBase.cc
  physics/grid/CalculatorTestBase.cc
)

target_link_libraries(CeleritasPhysicsTest PRIVATE celeritas CeleritasTest)

celeritas_setup_tests(SERIAL PREFIX physics/base
   LINK_LIBRARIES celeritas_final CeleritasPhysicsTest)
celeritas_add_test(physics/base/CutoffParams.test.cc
   LINK_LIBRARIES celeritas_final Celeritas::ROOT)
celeritas_cudaoptional_test(physics/base/Particle 
   LINK_LIBRARIES celeritas_final Celeritas::ROOT)
celeritas_cudaoptional_test(physics/base/Physics)
celeritas_add_test(physics/base/PhysicsStepUtils.test.cc)

celeritas_setup_tests(SERIAL PREFIX physics/grid
   LINK_LIBRARIES celeritas_final CeleritasPhysicsTest)
celeritas_add_test(physics/grid/GenericXsCalculator.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/GridIdFinder.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/Interpolator.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/InverseRangeCalculator.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/NonuniformGrid.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/RangeCalculator.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/TwodGridCalculator.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/UniformGrid.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/ValueGridBuilder.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/ValueGridInserter.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/grid/XsCalculator.test.cc LINK_LIBRARIES celeritas_final)

celeritas_setup_tests(SERIAL PREFIX physics/material
   LINK_LIBRARIES celeritas_final CeleritasPhysicsTest)
celeritas_add_test(physics/material/ElementSelector.test.cc LINK_LIBRARIES celeritas_final)
celeritas_cudaoptional_test(physics/material/Material
   LINK_LIBRARIES celeritas_final Celeritas::ROOT)

#-------------------------------------#
# Physics (EM)

celeritas_setup_tests(SERIAL PREFIX physics/em
  LINK_LIBRARIES CeleritasPhysicsTest)

celeritas_add_test(physics/em/BetheHeitler.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/EPlusGG.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/KleinNishina.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/LivermorePE.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/MollerBhabha.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/Rayleigh.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(physics/em/SeltzerBerger.test.cc LINK_LIBRARIES celeritas_final)

celeritas_add_test(physics/em/ImportedProcesses.test.cc ${_needs_root}
  ${_optional_geant4}
  GPU # TODO: delete when device pointers are no longer directly accessed
  LINK_LIBRARIES Celeritas::ROOT celeritas_final)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_cudaoptional_test(random/RngEngine LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/Selector.test.cc LINK_LIBRARIES celeritas_final)

celeritas_add_test(random/distributions/BernoulliDistribution.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/distributions/ExponentialDistribution.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/distributions/IsotropicDistribution.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/distributions/RadialDistribution.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/distributions/ReciprocalDistribution.test.cc LINK_LIBRARIES celeritas_final)
celeritas_add_test(random/distributions/UniformRealDistribution.test.cc LINK_LIBRARIES celeritas_final)


#-----------------------------------------------------------------------------#
# Sim

celeritas_setup_tests(SERIAL PREFIX sim)
if(CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  celeritas_add_test(sim/TrackInitializerStore.test.cc GPU
    SOURCES sim/TrackInitializerStore.test.cu
    LINK_LIBRARIES celeritas_cuda VecGeom::vecgeomcuda VecGeom::vgdml)
  get_property(vecgeom_static_target_location TARGET VecGeom::vecgeomcuda_static PROPERTY LOCATION)
  target_link_options(sim_TrackInitializerStore
    PRIVATE
    $<DEVICE_LINK:${vecgeom_static_target_location}>
    $<DEVICE_LINK:$<TARGET_FILE:celeritas_static>>
    #$<DEVICE_LINK:$<TARGET_FILE:${target}_static>>
  )
endif()

