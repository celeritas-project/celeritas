#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#
# CORE TEST LIBRARY
#-----------------------------------------------------------------------------#

set(CELERITAS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
configure_file(testdetail/TestConfig.cc.in TestConfig.cc @ONLY)

set(CELER_SOURCES)
list(APPEND CELER_SOURCES
  Test.cc
  testdetail/TestMacrosImpl.cc
  testdetail/NonMasterResultPrinter.cc
  testdetail/ParallelHandler.cc
  testdetail/TestMain.cc
  "${CMAKE_CURRENT_BINARY_DIR}/TestConfig.cc"
)

add_library(CelerTest ${CELER_SOURCES})
add_library(Celeritas::Test ALIAS CelerTest)
set_property(TARGET CelerTest PROPERTY CXX_STANDARD 14)
celeritas_target_link_libraries(CelerTest PRIVATE celeritas PUBLIC GTest::GTest)

target_include_directories(CelerTest
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
    celeritas/geo/HeuristicGeoTestBase.cu
    celeritas/mat/Material.test.cu
    celeritas/random/RngEngine.test.cu
    celeritas/random/curand/CurandPerformance.test.cu
    celeritas/track/TrackInit.test.cu
    PROPERTIES LANGUAGE HIP
  )
endif()

#-----------------------------------------------------------------------------#
# TEST OPTIONS
#-----------------------------------------------------------------------------#

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
  set(_geant4_test_env)
  foreach(_ds G4EMLOW G4ENSDFSTATE)
    list(APPEND _geant4_test_env
      "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}"
    )
  endforeach()
  set(_optional_geant4_env ENVIRONMENT "${_geant4_test_env}")
  set(_optional_geant4_link
    LINK_LIBRARIES ${Geant4_LIBRARIES}
  )
  set(_needs_geant4 ${_optional_geant4_env})
endif()

#-----------------------------------------------------------------------------#
# TESTING ADDITION TESTS
#-----------------------------------------------------------------------------#
celeritas_setup_tests(SERIAL PREFIX detail)

celeritas_add_test(TestMacros.test.cc)

#-----------------------------------------------------------------------------#
# CORECEL TESTS
#-----------------------------------------------------------------------------#
celeritas_setup_tests(SERIAL PREFIX corecel)

# Cont
celeritas_add_test(corecel/cont/Array.test.cc)
celeritas_add_test(corecel/cont/Span.test.cc)
celeritas_add_test(corecel/cont/LabelIdMultiMap.test.cc)
celeritas_device_test(corecel/cont/Range)

# Data
celeritas_device_test(corecel/data/Collection)
celeritas_add_test(corecel/data/Copier.test.cc GPU)
celeritas_add_test(corecel/data/DeviceAllocation.test.cc GPU)
celeritas_add_test(corecel/data/DeviceVector.test.cc GPU)
celeritas_add_test(corecel/OpaqueId.test.cc)
celeritas_device_test(corecel/data/StackAllocator)

# IO
celeritas_add_test(corecel/io/Join.test.cc)
celeritas_add_test(corecel/io/Logger.test.cc)
celeritas_add_test(corecel/io/OutputManager.test.cc)
celeritas_add_test(corecel/io/Repr.test.cc)
celeritas_add_test(corecel/io/StringEnumMap.test.cc)

# Math
celeritas_add_test(corecel/math/Algorithms.test.cc)
celeritas_add_test(corecel/math/ArrayUtils.test.cc)
celeritas_add_test(corecel/math/HashUtils.test.cc)
celeritas_device_test(corecel/math/NumericLimits)
celeritas_add_test(corecel/math/Quantity.test.cc)
celeritas_add_test(corecel/math/SoftEqual.test.cc)
celeritas_add_test(corecel/math/VectorUtils.test.cc)

# Sys
celeritas_add_test(corecel/sys/Environment.test.cc
  ENVIRONMENT "ENVTEST_ONE=1;ENVTEST_ZERO=0;ENVTEST_EMPTY="
)
celeritas_add_test(corecel/sys/TypeDemangler.test.cc)
celeritas_add_test(corecel/sys/ScopedSignalHandler.test.cc)
celeritas_add_test(corecel/sys/ScopedStreamRedirect.test.cc)
celeritas_add_test(corecel/sys/Stopwatch.test.cc ADDED_TESTS _stopwatch)
set_tests_properties(${_stopwatch} PROPERTIES LABELS "nomemcheck")

#-----------------------------------------------------------------------------#
# ORANGE TESTS
#-----------------------------------------------------------------------------#
celeritas_setup_tests(SERIAL PREFIX orange)

# Base-level
celeritas_add_test(orange/BoundingBox.test.cc)

# Construction
celeritas_add_test(orange/construct/SurfaceInserter.test.cc)
celeritas_add_test(orange/construct/VolumeInserter.test.cc)

# Surfaces
celeritas_add_test(orange/surf/detail/QuadraticSolver.test.cc)
celeritas_add_test(orange/surf/CylCentered.test.cc)
celeritas_add_test(orange/surf/GeneralQuadric.test.cc)
celeritas_add_test(orange/surf/PlaneAligned.test.cc)
celeritas_add_test(orange/surf/Sphere.test.cc)
celeritas_add_test(orange/surf/SphereCentered.test.cc)
celeritas_device_test(orange/surf/SurfaceAction)

# Universe details
celeritas_add_test(orange/univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(orange/univ/detail/LogicStack.test.cc)
celeritas_add_test(orange/univ/detail/SurfaceFunctors.test.cc)

### Tests that require a geometry input ###

add_library(CeleritasOrangeTest
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
# CELERITAS TESTS
#-----------------------------------------------------------------------------#
add_library(CeleritasTest
  celeritas/grid/CalculatorTestBase.cc
  celeritas/phys/InteractionIO.cc
  celeritas/phys/InteractorHostTestBase.cc
  celeritas/phys/SecondaryIO.cc
  celeritas/phys/MockModel.cc
  celeritas/phys/MockProcess.cc
  celeritas/GeantTestBase.cc
  celeritas/GlobalTestBase.cc
  celeritas/GlobalGeoTestBase.cc
  celeritas/MockTestBase.cc
  celeritas/SimpleTestBase.cc
  celeritas/global/AlongStepTestBase.cc
  celeritas/global/StepperTestBase.cc
)
celeritas_target_link_libraries(CeleritasTest
  PRIVATE celeritas celeritas_io CelerTest)

#-----------------------------------------------------------------------------#
celeritas_setup_tests(SERIAL PREFIX celeritas LINK_LIBRARIES CeleritasTest)

#-------------------------------------#
# (Root level tests)
celeritas_add_test(celeritas/Constants.test.cc ${_optional_geant4_link})

#-------------------------------------#
# EM
set(CELERITASTEST_PREFIX celeritas/em)
celeritas_add_test(celeritas/em/BetheHeitler.test.cc)
celeritas_add_test(celeritas/em/CombinedBrem.test.cc)
celeritas_add_test(celeritas/em/EPlusGG.test.cc)
celeritas_add_test(celeritas/em/Fluctuation.test.cc)
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

#-------------------------------------#
# External
set(CELERITASTEST_PREFIX celeritas/ext)

if(CELERITAS_USE_VecGeom)
  celeritas_device_test(celeritas/ext/Vecgeom
    LINK_LIBRARIES VecGeom::vecgeom)
endif()

celeritas_add_test(celeritas/ext/EventReader.test.cc ${_needs_hepmc})
celeritas_add_test(celeritas/ext/GeantImporter.test.cc ${_needs_geant4}
  LINK_LIBRARIES Celeritas::IO)
celeritas_add_test(celeritas/ext/MpiCommunicator.test.cc
  NP ${CELERITASTEST_NP_DEFAULT})
celeritas_add_test(celeritas/ext/RootImporter.test.cc ${_needs_root}
  LINK_LIBRARIES Celeritas::IO)

#-------------------------------------#
# Field
set(CELERITASTEST_PREFIX celeritas/field)

celeritas_add_test(celeritas/field/Fields.test.cc
SOURCES
  celeritas/field/detail/MagFieldMap.cc
  celeritas/field/detail/CMSFieldMapReader.cc
)
celeritas_add_test(celeritas/field/Steppers.test.cc)
celeritas_add_test(celeritas/field/FieldDriver.test.cc)
celeritas_add_test(celeritas/field/FieldPropagator.test.cc ${_needs_geo})
celeritas_add_test(celeritas/field/LinearPropagator.test.cc ${_needs_geo})

#-------------------------------------#
# Geo
set(CELERITASTEST_PREFIX celeritas/geo)

set(_geo_args GPU ${_needs_geo}
  SOURCES celeritas/geo/HeuristicGeoTestBase.cc)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _geo_args celeritas/geo/HeuristicGeoTestBase.cu)
endif()
if(CELERITAS_USE_VecGeom)
  list(APPEND _geo_args LINK_LIBRARIES VecGeom::vecgeom)
else()
  # FIXME: backscattering (or field direction change) into volume crashes ORANGE
  list(APPEND _geo_args DISABLE)
endif()
celeritas_add_test(celeritas/geo/Geometry.test.cc ${_geo_args})

celeritas_add_test(celeritas/geo/GeoMaterial.test.cc
  ${_needs_root} ${_needs_geo}
  LINK_LIBRARIES Celeritas::IO)

#-------------------------------------#
# Global
set(CELERITASTEST_PREFIX celeritas/global)
celeritas_add_test(celeritas/global/ActionManager.test.cc)
celeritas_add_test(celeritas/global/AlongStep.test.cc
  ${_optional_geant4_env})
celeritas_add_test(celeritas/global/Stepper.test.cc
  GPU ${_needs_geant4}
  FILTER
    # NOTE: these can be run in the same invocation once Geant4 reload is fixed
    "TestEm3*"
    "TestEm15FieldTest.*"
)

#-------------------------------------#
# Grid
set(CELERITASTEST_PREFIX celeritas/grid)
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

#-------------------------------------#
# IO
set(CELERITASTEST_PREFIX celeritas/io)
celeritas_add_test(celeritas/io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-------------------------------------#
# Mat
set(CELERITASTEST_PREFIX celeritas/mat)
celeritas_add_test(celeritas/mat/ElementSelector.test.cc)
celeritas_device_test(celeritas/mat/Material
  LINK_LIBRARIES Celeritas::IO)

#-------------------------------------#
# Phys
set(CELERITASTEST_PREFIX celeritas/phys)
celeritas_add_test(celeritas/phys/CutoffParams.test.cc
  LINK_LIBRARIES Celeritas::IO)
celeritas_device_test(celeritas/phys/Particle
  LINK_LIBRARIES Celeritas::IO)
celeritas_device_test(celeritas/phys/Physics)
celeritas_add_test(celeritas/phys/PhysicsStepUtils.test.cc)
celeritas_add_test(celeritas/phys/PrimaryGenerator.test.cc)

#-----------------------------------------------------------------------------#
# Random
set(CELERITASTEST_PREFIX celeritas/random)

celeritas_device_test(celeritas/random/RngEngine)
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

if(CELERITAS_USE_CUDA)
  celeritas_add_test(celeritas/random/curand/CurandPerformance.test.cc GPU
    SOURCES celeritas/random/curand/CurandPerformance.test.cu
    LINK_LIBRARIES Celeritas::DeviceToolkit
    ${_disable_if_debug}
  )
endif()

#-------------------------------------#
# Track
set(CELERITASTEST_PREFIX celeritas/track)
celeritas_device_test(celeritas/track/TrackInit ${_needs_cuda})

#-----------------------------------------------------------------------------#
# DATA UPDATE
#-----------------------------------------------------------------------------#

set(_fourslabs "${CMAKE_CURRENT_SOURCE_DIR}/celeritas/data/four-steel-slabs")
set(_cmake_env)
if(CELERITAS_USE_Geant4)
  set(_cmake_env
    "${CMAKE_COMMAND}" "-E" "env" "${_geant4_test_env}"
  )
endif()
add_custom_target(update-four-slabs
  COMMAND
    ${_cmake_env}
    "$<TARGET_FILE:celer-export-geant>"
    "${_fourslabs}.gdml"
    "${_fourslabs}.geant.json"
    "${_fourslabs}.root"
  VERBATIM
)
add_dependencies(update-four-slabs celer-export-geant)

