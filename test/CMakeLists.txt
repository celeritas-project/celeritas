#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# TEST LIBRARY
#-----------------------------------------------------------------------------#

set(CELERITAS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
configure_file(detail/TestConfig.cc.in TestConfig.cc @ONLY)

set(CELER_SOURCES)
list(APPEND CELER_SOURCES
  Test.cc
  detail/Macros.cc
  detail/NonMasterResultPrinter.cc
  detail/ParallelHandler.cc
  detail/TestMain.cc
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
# HIP SUPPORT
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_HIP)
  set_source_files_properties(
    corecel/data/Collection.test.cu
    corecel/math/NumericLimits.test.cu
    corecel/cont/Range.test.cu
    corecel/data/StackAllocator.test.cu
    celeritas/field/FieldDriver.test.cu
    celeritas/field/FieldPropagator.test.cu
    celeritas/field/Steppers.test.cu
    celeritas/field/UserMapField.test.cu
    celeritas/field/UserParamField.test.cu
    orange/surf/SurfaceAction.test.cu
    orange/univ/SimpleUnitTracker.test.cu
    celeritas/phys/Particle.test.cu
    celeritas/phys/Physics.test.cu
    celeritas/mat/Material.test.cu
    celeritas/random/RngEngine.test.cu
    celeritas/random/curand/CurandPerformance.test.cu
    celeritas/track/TrackInit.test.cu
    PROPERTIES LANGUAGE HIP
  )
endif()

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

function(celeritas_device_test base)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
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
if(CELERITAS_DEBUG)
  set(_disable_if_debug DISABLE)
endif()
if(NOT CELERITAS_USE_VecGeom AND NOT CELERITAS_USE_JSON)
  set(_needs_geo DISABLE)
endif()

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
else()
  # Optional dependence on low-energy EM data
  set(_ds G4EMLOW)
  set(_optional_geant4_env
     ENVIRONMENT "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}"
  )
  set(_optional_geant4_link
    LINK_LIBRARIES ${Geant4_LIBRARIES}
  )
  set(_needs_geant4 ${_optional_geant4_env})
endif()

#-----------------------------------------------------------------------------#
# Gtest additions
celeritas_setup_tests(SERIAL PREFIX gtest/detail)

celeritas_add_test(detail/test/Macros.test.cc)

#-----------------------------------------------------------------------------#
# Base

celeritas_setup_tests(SERIAL PREFIX base)

celeritas_add_test(corecel/math/Algorithms.test.cc)
celeritas_add_test(corecel/cont/Array.test.cc)
celeritas_add_test(corecel/math/ArrayUtils.test.cc)
celeritas_add_test(celeritas/Constants.test.cc ${_optional_geant4_link})
celeritas_add_test(corecel/data/Copier.test.cc GPU)
celeritas_add_test(corecel/data/DeviceAllocation.test.cc GPU)
celeritas_add_test(corecel/data/DeviceVector.test.cc GPU)
celeritas_add_test(corecel/io/Join.test.cc)
celeritas_add_test(corecel/data/OpaqueId.test.cc)
celeritas_add_test(corecel/math/Quantity.test.cc)
celeritas_add_test(corecel/io/Repr.test.cc)
celeritas_add_test(corecel/sys/ScopedStreamRedirect.test.cc)
celeritas_add_test(corecel/math/SoftEqual.test.cc)
celeritas_add_test(corecel/cont/Span.test.cc)
celeritas_add_test(corecel/sys/TypeDemangler.test.cc)
celeritas_add_test(corecel/math/VectorUtils.test.cc)

celeritas_add_test(corecel/sys/Stopwatch.test.cc ADDED_TESTS _test_name)
set_tests_properties(${_test_name} PROPERTIES LABELS "nomemcheck")

celeritas_device_test(corecel/data/Collection)
celeritas_device_test(corecel/math/NumericLimits)
celeritas_device_test(corecel/cont/Range)
celeritas_device_test(corecel/data/StackAllocator)

#-----------------------------------------------------------------------------#
# Comm

celeritas_setup_tests(PREFIX comm)

celeritas_add_test(celeritas/ext/Communicator.test.cc)
celeritas_add_test(corecel/sys/Environment.test.cc
  ENVIRONMENT "ENVTEST_ONE=1;ENVTEST_ZERO=0;ENVTEST_EMPTY="
)
celeritas_add_test(corecel/io/Logger.test.cc)
celeritas_add_test(corecel/sys/ScopedSignalHandler.test.cc)

#-----------------------------------------------------------------------------#
# Field

celeritas_setup_tests(SERIAL PREFIX field)

celeritas_device_test(celeritas/field/Steppers)
celeritas_device_test(celeritas/field/FieldDriver)
celeritas_device_test(celeritas/field/FieldPropagator ${_needs_geo})

### User field ###

celeritas_device_test(celeritas/field/UserParamField ${_needs_geo})
celeritas_device_test(celeritas/field/UserMapField
  SOURCES
    celeritas/field/detail/MagFieldMap.cc
    celeritas/field/detail/CMSFieldMapReader.cc
  ${_needs_geo}
)

#-----------------------------------------------------------------------------#
# VecGeom

celeritas_setup_tests(SERIAL PREFIX vecgeom
  LINK_LIBRARIES ${_vgcudatest} VecGeom::vecgeom)

if(CELERITAS_USE_VecGeom)
  celeritas_device_test(celeritas/ext/Vecgeom)
endif()

#-----------------------------------------------------------------------------#
# Implementation-independent geometry

celeritas_setup_tests(SERIAL PREFIX geometry)

celeritas_add_test(orange/BoundingBox.test.cc)

celeritas_add_test(celeritas/geo/GeoMaterial.test.cc ${_needs_root} ${_needs_geo}
  LINK_LIBRARIES Celeritas::IO)
celeritas_add_test(celeritas/field/LinearPropagator.test.cc ${_needs_geo})

#-----------------------------------------------------------------------------#
# ORANGE geometry

celeritas_setup_tests(SERIAL PREFIX orange)

celeritas_add_test(orange/construct/SurfaceInserter.test.cc)
celeritas_add_test(orange/construct/VolumeInserter.test.cc)

celeritas_add_test(orange/surf/detail/QuadraticSolver.test.cc)
celeritas_add_test(orange/surf/CylCentered.test.cc)
celeritas_add_test(orange/surf/GeneralQuadric.test.cc)
celeritas_add_test(orange/surf/PlaneAligned.test.cc)
celeritas_add_test(orange/surf/Sphere.test.cc)
celeritas_add_test(orange/surf/SphereCentered.test.cc)

celeritas_device_test(orange/surf/SurfaceAction)

celeritas_add_test(orange/univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(orange/univ/detail/LogicStack.test.cc)
celeritas_add_test(orange/univ/detail/SurfaceFunctors.test.cc)

### Tests that require a geometry input ###

celeritas_add_library(CeleritasOrangeTest
  orange/OrangeGeoTestBase.cc
)
celeritas_target_link_libraries(CeleritasOrangeTest
  PRIVATE Celeritas::Test
  PUBLIC Celeritas::Core
)
set(CELERITASTEST_LINK_LIBRARIES CeleritasOrangeTest)

celeritas_add_test(orange/univ/detail/SenseCalculator.test.cc)
celeritas_add_test(orange/univ/VolumeView.test.cc)
celeritas_device_test(orange/univ/SimpleUnitTracker)
celeritas_add_test(orange/Orange.test.cc)

#-----------------------------------------------------------------------------#
# I/O (ROOT)

celeritas_setup_tests(SERIAL PREFIX io)

celeritas_add_test(celeritas/ext/RootImporter.test.cc ${_needs_root}
  LINK_LIBRARIES Celeritas::IO)
celeritas_add_test(celeritas/ext/EventReader.test.cc ${_needs_hepmc})
celeritas_add_test(celeritas/io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-----------------------------------------------------------------------------#
# Physics

add_library(CeleritasPhysicsTest
  celeritas/phys/InteractionIO.cc
  celeritas/phys/InteractorHostTestBase.cc
  celeritas/phys/SecondaryIO.cc
  celeritas/phys/MockModel.cc
  celeritas/phys/MockProcess.cc
  celeritas/phys/PhysicsTestBase.cc
  celeritas/grid/CalculatorTestBase.cc
)

celeritas_target_link_libraries(CeleritasPhysicsTest PRIVATE celeritas CeleritasTest)

celeritas_setup_tests(SERIAL PREFIX physics/base
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(celeritas/phys/CutoffParams.test.cc
  LINK_LIBRARIES Celeritas::IO)
celeritas_device_test(celeritas/phys/Particle
  LINK_LIBRARIES Celeritas::IO)
celeritas_device_test(celeritas/phys/Physics)
celeritas_add_test(celeritas/phys/PhysicsStepUtils.test.cc)

#-------------------------------------#
# Physics (grid)

celeritas_setup_tests(SERIAL PREFIX physics/grid
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(celeritas/grid/GenericXsCalculator.test.cc)
celeritas_add_test(celeritas/grid/GridIdFinder.test.cc)
celeritas_add_test(celeritas/grid/Interpolator.test.cc)
celeritas_add_test(celeritas/grid/InverseRangeCalculator.test.cc)
celeritas_add_test(celeritas/grid/NonuniformGrid.test.cc)
celeritas_add_test(celeritas/grid/PolyEvaluator.test.cc)
celeritas_add_test(celeritas/grid/RangeCalculator.test.cc)
celeritas_add_test(celeritas/grid/TwodGridCalculator.test.cc)
celeritas_add_test(celeritas/grid/UniformGrid.test.cc)
celeritas_add_test(celeritas/grid/ValueGridBuilder.test.cc)
celeritas_add_test(celeritas/grid/ValueGridInserter.test.cc)
celeritas_add_test(celeritas/grid/XsCalculator.test.cc)

celeritas_setup_tests(SERIAL PREFIX physics/material
  LINK_LIBRARIES CeleritasPhysicsTest)
celeritas_add_test(celeritas/mat/ElementSelector.test.cc)
celeritas_device_test(celeritas/mat/Material
  LINK_LIBRARIES Celeritas::IO)

#-------------------------------------#
# Physics (EM)

celeritas_setup_tests(SERIAL PREFIX physics/em
  LINK_LIBRARIES CeleritasPhysicsTest)

celeritas_add_test(celeritas/em/BetheHeitler.test.cc)
celeritas_add_test(celeritas/em/CombinedBrem.test.cc)
celeritas_add_test(celeritas/em/EnergyLossFluctuation.test.cc)
celeritas_add_test(celeritas/em/EPlusGG.test.cc)
celeritas_add_test(celeritas/em/KleinNishina.test.cc)
celeritas_add_test(celeritas/em/LivermorePE.test.cc)
celeritas_add_test(celeritas/em/MollerBhabha.test.cc)
celeritas_add_test(celeritas/em/MuBremsstrahlung.test.cc)
celeritas_add_test(celeritas/em/Rayleigh.test.cc)
celeritas_add_test(celeritas/em/RelativisticBrem.test.cc)
celeritas_add_test(celeritas/em/SeltzerBerger.test.cc)
celeritas_add_test(celeritas/em/TsaiUrbanDistribution.test.cc)

celeritas_add_test(celeritas/em/ImportedProcesses.test.cc ${_needs_root}
  ${_optional_geant4_env}
  LINK_LIBRARIES Celeritas::IO)

celeritas_add_test(celeritas/em/UrbanMsc.test.cc ${_needs_root}
  ${_optional_geant4_env}
  LINK_LIBRARIES Celeritas::IO)

#-----------------------------------------------------------------------------#
# Random

celeritas_setup_tests(SERIAL PREFIX random)

celeritas_device_test(celeritas/random/RngEngine)
if(CELERITAS_USE_CUDA)
  celeritas_add_test(celeritas/random/curand/CurandPerformance.test.cc GPU
    SOURCES celeritas/random/curand/CurandPerformance.test.cu
    LINK_LIBRARIES CUDA::cudart
    ${_disable_if_debug}
  )
endif()
celeritas_add_test(celeritas/random/Selector.test.cc)
celeritas_add_test(celeritas/random/XorwowRngEngine.test.cc GPU)

celeritas_add_test(celeritas/random/distribution/BernoulliDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/ExponentialDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/GammaDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/IsotropicDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/NormalDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/PoissonDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/RadialDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/ReciprocalDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/UniformBoxDistribution.test.cc)
celeritas_add_test(celeritas/random/distribution/UniformRealDistribution.test.cc)

#-----------------------------------------------------------------------------#
# Sim

celeritas_setup_tests(SERIAL PREFIX sim)

celeritas_add_test(celeritas/global/ActionManager.test.cc)
celeritas_add_test(corecel/io/OutputManager.test.cc)
celeritas_device_test(celeritas/track/TrackInit ${_needs_cuda})
