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
celeritas_target_link_libraries(CeleritasTest PRIVATE celeritas PUBLIC GTest::GTest)

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
celeritas_add_test(base/Join.test.cc)
celeritas_add_test(base/OpaqueId.test.cc)
celeritas_add_test(base/Quantity.test.cc)
celeritas_add_test(base/ScopedStreamRedirect.test.cc)
celeritas_add_test(base/SoftEqual.test.cc)
celeritas_add_test(base/Span.test.cc)
celeritas_add_test(base/SpanRemapper.test.cc)
celeritas_add_test(base/Stopwatch.test.cc)
celeritas_add_test(base/TypeDemangler.test.cc)
celeritas_add_test(base/VectorUtils.test.cc)

celeritas_cudaoptional_test(base/Collection)
celeritas_cudaoptional_test(base/NumericLimits)
celeritas_cudaoptional_test(base/Range)
celeritas_cudaoptional_test(base/StackAllocator)

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)

celeritas_add_test(comm/Communicator.test.cc)
celeritas_add_test(comm/Logger.test.cc)


#-----------------------------------------------------------------------------#
# Field

celeritas_setup_tests(SERIAL PREFIX field)

celeritas_cudaoptional_test(field/RungeKutta)
celeritas_cudaoptional_test(field/FieldDriver)

if(CELERITAS_USE_VecGeom)
  if(CELERITAS_USE_CUDA)
    celeritas_add_library(CeleritasPropagatorTest field/FieldPropagator.test.cu)
    celeritas_target_link_libraries(CeleritasPropagatorTest
      PRIVATE celeritas VecGeom::vecgeom
    )
    list(APPEND CELERITASTEST_LINK_LIBRARIES CeleritasPropagatorTest)
    set(_CeleritasPropagatorTest CeleritasPropagatorTest)
  endif()

  celeritas_add_test(field/FieldPropagator.test.cc GPU
    LINK_LIBRARIES Celeritas::Core VecGeom::vecgeom ${_CeleritasPropagatorTest})
endif()

#-----------------------------------------------------------------------------#
# Field - User field

celeritas_cudaoptional_test(field/UserParamField)

set(CELER_USERFIELD_SOURCES
  field/detail/MagFieldMap.cc
  field/detail/CMSFieldMapReader.cc
)
if(CELERITAS_USE_CUDA)
  list(APPEND CELER_USERFIELD_SOURCES
    field/UserMapField.test.cu
  )
endif()
celeritas_add_library(CeleritasUserFieldTest ${CELER_USERFIELD_SOURCES})
celeritas_target_link_libraries(CeleritasUserFieldTest
  PUBLIC Celeritas::Core
)

celeritas_setup_tests(SERIAL PREFIX fieldmap
  LINK_LIBRARIES CeleritasUserFieldTest)
celeritas_add_test(field/UserMapField.test.cc GPU)

#-----------------------------------------------------------------------------#
# Geometry

if(CELERITAS_USE_VecGeom)
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
  celeritas_target_link_libraries(CeleritasVecgeomTest
    PRIVATE Celeritas::Test VecGeom::vecgeom
    PUBLIC Celeritas::Core
  )

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
  LINK_LIBRARIES Celeritas::ROOT)
celeritas_add_test(io/EventReader.test.cc ${_needs_hepmc})
celeritas_add_test(io/SeltzerBergerReader.test.cc ${_needs_geant4})

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

celeritas_target_link_libraries(CeleritasPhysicsTest PRIVATE celeritas CeleritasTest)

celeritas_setup_tests(SERIAL PREFIX physics/base
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(physics/base/CutoffParams.test.cc
  LINK_LIBRARIES Celeritas::ROOT)
celeritas_cudaoptional_test(physics/base/Particle
  LINK_LIBRARIES Celeritas::ROOT)
celeritas_cudaoptional_test(physics/base/Physics)
celeritas_add_test(physics/base/PhysicsStepUtils.test.cc)

celeritas_setup_tests(SERIAL PREFIX physics/grid
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(physics/grid/GenericXsCalculator.test.cc)
celeritas_add_test(physics/grid/GridIdFinder.test.cc)
celeritas_add_test(physics/grid/Interpolator.test.cc)
celeritas_add_test(physics/grid/InverseRangeCalculator.test.cc)
celeritas_add_test(physics/grid/NonuniformGrid.test.cc)
celeritas_add_test(physics/grid/RangeCalculator.test.cc)
celeritas_add_test(physics/grid/TwodGridCalculator.test.cc)
celeritas_add_test(physics/grid/UniformGrid.test.cc)
celeritas_add_test(physics/grid/ValueGridBuilder.test.cc)
celeritas_add_test(physics/grid/ValueGridInserter.test.cc)
celeritas_add_test(physics/grid/XsCalculator.test.cc)

celeritas_setup_tests(SERIAL PREFIX physics/material
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(physics/material/ElementSelector.test.cc)
celeritas_cudaoptional_test(physics/material/Material
  LINK_LIBRARIES Celeritas::ROOT)

#-------------------------------------#
# Physics (EM)

celeritas_setup_tests(SERIAL PREFIX physics/em
  LINK_LIBRARIES CeleritasPhysicsTest)

celeritas_add_test(physics/em/BetheHeitler.test.cc)
celeritas_add_test(physics/em/EnergyLossDistribution.test.cc)
celeritas_add_test(physics/em/EPlusGG.test.cc)
celeritas_add_test(physics/em/KleinNishina.test.cc)
celeritas_add_test(physics/em/LivermorePE.test.cc)
celeritas_add_test(physics/em/MollerBhabha.test.cc)
celeritas_add_test(physics/em/MuBremsstrahlung.test.cc)
celeritas_add_test(physics/em/Rayleigh.test.cc)
celeritas_add_test(physics/em/SeltzerBerger.test.cc)
celeritas_add_test(physics/em/TsaiUrbanDistribution.test.cc)

celeritas_add_test(physics/em/ImportedProcesses.test.cc ${_needs_root}
  ${_optional_geant4}
  GPU # TODO: delete when device pointers are no longer directly accessed
  LINK_LIBRARIES Celeritas::ROOT)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_cudaoptional_test(random/RngEngine)
celeritas_cudaoptional_test(random/curand/CurandPerformance)
celeritas_add_test(random/Selector.test.cc)

celeritas_add_test(random/distributions/BernoulliDistribution.test.cc)
celeritas_add_test(random/distributions/ExponentialDistribution.test.cc)
celeritas_add_test(random/distributions/GammaDistribution.test.cc)
celeritas_add_test(random/distributions/IsotropicDistribution.test.cc)
celeritas_add_test(random/distributions/NormalDistribution.test.cc)
celeritas_add_test(random/distributions/PoissonDistribution.test.cc)
celeritas_add_test(random/distributions/RadialDistribution.test.cc)
celeritas_add_test(random/distributions/ReciprocalDistribution.test.cc)
celeritas_add_test(random/distributions/UniformBoxDistribution.test.cc)
celeritas_add_test(random/distributions/UniformRealDistribution.test.cc)

#-----------------------------------------------------------------------------#
# Sim

celeritas_setup_tests(SERIAL PREFIX sim)
if(CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  celeritas_add_test(sim/TrackInit.test.cc GPU
    SOURCES sim/TrackInit.test.cu
    LINK_LIBRARIES VecGeom::vecgeom)
endif()

