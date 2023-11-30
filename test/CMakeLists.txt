#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#
# TEST HARNESS
#-----------------------------------------------------------------------------#

file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" CELERITAS_SOURCE_DIR)
configure_file(celeritas_test_config.h.in celeritas_test_config.h @ONLY)

celeritas_add_library(testcel_harness
  Test.cc
  testdetail/NonMasterResultPrinter.cc
  testdetail/TestMacrosImpl.cc
  testdetail/TestMainImpl.cc
)
target_compile_features(testcel_harness PUBLIC cxx_std_14)
celeritas_target_link_libraries(testcel_harness
  PUBLIC Celeritas::corecel GTest::GTest
  PRIVATE ${nlohmann_json_LIBRARIES}
)
celeritas_target_include_directories(testcel_harness
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

#-----------------------------------------------------------------------------#
# HIP SUPPORT
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_HIP)
  file(GLOB_RECURSE _cufiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
  set_source_files_properties(${_cufiles}
    PROPERTIES LANGUAGE HIP
  )
endif()

#-----------------------------------------------------------------------------#
# TEST OPTIONS
#-----------------------------------------------------------------------------#

function(celeritas_add_device_test base)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    set(_cuda_args GPU SOURCES "${base}.test.cu")
  endif()
  celeritas_add_test("${base}.test.cc" ${_cuda_args} ${ARGN})
endfunction()

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()
if(NOT CELERITAS_USE_JSON)
  set(_needs_json DISABLE)
endif()
if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()
if(CELERITAS_USE_Geant4)
  # Optional dependence on low-energy EM data
  set(_geant4_test_env)
  foreach(_ds G4EMLOW G4ENSDFSTATE)
    list(APPEND _geant4_test_env
      "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}"
    )
  endforeach()
  set(_optional_geant4_env ENVIRONMENT "${_geant4_test_env}")
  set(_needs_geant4 ${_optional_geant4_env})
else()
  set(_needs_geant4 DISABLE)
endif()
if(CELERITAS_DEBUG)
  set(_disable_if_debug DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE" AND NOT CELERITAS_USE_JSON)
  set(_needs_geo DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  set(_core_geo_lib VecGeom::vecgeom)
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  set(_core_geo_lib ${Geant4_LIBRARIES})
else()
  set(_core_geo_lib Celeritas::orange)
endif()
set(_geo_libs
  Celeritas::orange ${VecGeom_LIBRARIES} ${Geant4_LIBRARIES}
)

if(CELERITAS_REAL_TYPE STREQUAL "double")
  set(_needs_double)
  set(_fixme_single)
else()
  # Test relies on "gold" data; don't bother updating
  set(_needs_double DISABLE)
  # Test has legitimate errors and we should fix if we support
  # single-precision in the long term
  set(_fixme_single DISABLE)
endif()

#-----------------------------------------------------------------------------#
# GOOGLETEST EXTENSION TESTS
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL PREFIX testdetail)

celeritas_add_test(TestMacros.test.cc)

#-----------------------------------------------------------------------------#
# CORECEL TESTS
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_core
  corecel/ScopedLogStorer.cc
)
celeritas_target_link_libraries(testcel_core
  PRIVATE Celeritas::testcel_harness Celeritas::corecel
)

celeritas_setup_tests(SERIAL PREFIX corecel
  LINK_LIBRARIES testcel_core Celeritas::corecel
)

celeritas_add_test(corecel/OpaqueId.test.cc)

# Cont
set(CELERITASTEST_PREFIX corecel/cont)
celeritas_add_test(corecel/cont/Array.test.cc)
celeritas_add_test(corecel/cont/InitializedValue.test.cc)
celeritas_add_test(corecel/cont/Span.test.cc)
celeritas_add_test(corecel/cont/LabelIdMultiMap.test.cc)
celeritas_add_test(corecel/cont/VariantUtils.test.cc)
celeritas_add_device_test(corecel/cont/Range)

# Data
set(CELERITASTEST_PREFIX corecel/data)
celeritas_add_device_test(corecel/data/Collection)
celeritas_add_test(corecel/data/Copier.test.cc GPU)
celeritas_add_test(corecel/data/DeviceAllocation.test.cc GPU)
celeritas_add_test(corecel/data/DeviceVector.test.cc GPU)
celeritas_add_device_test(corecel/data/ObserverPtr)
celeritas_add_test(corecel/data/LdgIterator.test.cc)
celeritas_add_test(corecel/data/HyperslabIndexer.test.cc)
celeritas_add_device_test(corecel/data/StackAllocator)

# Grid
set(CELERITASTEST_PREFIX corecel/grid)
celeritas_add_test(corecel/grid/Interpolator.test.cc)
celeritas_add_test(corecel/grid/NonuniformGrid.test.cc)
celeritas_add_test(corecel/grid/TwodGridCalculator.test.cc)
celeritas_add_test(corecel/grid/UniformGrid.test.cc)

# IO
set(CELERITASTEST_PREFIX corecel/io)
celeritas_add_test(corecel/io/EnumStringMapper.test.cc)
celeritas_add_test(corecel/io/Label.test.cc)
celeritas_add_test(corecel/io/Join.test.cc)
celeritas_add_test(corecel/io/Logger.test.cc)
celeritas_add_test(corecel/io/OutputRegistry.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(corecel/io/Repr.test.cc)
celeritas_add_test(corecel/io/StringEnumMapper.test.cc)
celeritas_add_test(corecel/io/StringUtils.test.cc)

# Math
set(CELERITASTEST_PREFIX corecel/math)
celeritas_add_test(corecel/math/Algorithms.test.cc)
celeritas_add_test(corecel/math/ArrayOperators.test.cc)
celeritas_add_test(corecel/math/ArrayUtils.test.cc)
celeritas_add_test(corecel/math/HashUtils.test.cc)
celeritas_add_device_test(corecel/math/NumericLimits)
celeritas_add_test(corecel/math/Quantity.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(corecel/math/SoftEqual.test.cc)

# Sys
set(CELERITASTEST_PREFIX corecel/sys)
celeritas_add_test(corecel/sys/Environment.test.cc
  ENVIRONMENT "ENVTEST_ONE=1;ENVTEST_ZERO=0;ENVTEST_EMPTY="
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
)
celeritas_add_test(corecel/sys/MpiCommunicator.test.cc
  NP ${CELERITASTEST_NP_DEFAULT})
celeritas_add_test(corecel/sys/MultiExceptionHandler.test.cc)
celeritas_add_test(corecel/sys/TypeDemangler.test.cc)
celeritas_add_test(corecel/sys/ScopedSignalHandler.test.cc)
celeritas_add_test(corecel/sys/ScopedStreamRedirect.test.cc)
celeritas_add_test(corecel/sys/Stopwatch.test.cc ADDED_TESTS _stopwatch)
set_tests_properties(${_stopwatch} PROPERTIES LABELS "nomemcheck")
celeritas_add_test(corecel/sys/Version.test.cc)

#-----------------------------------------------------------------------------#
# ORANGE TESTS
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_orange
  orange/OrangeGeoTestBase.cc
)
celeritas_target_link_libraries(testcel_orange
  PRIVATE Celeritas::testcel_harness Celeritas::testcel_core Celeritas::orange
)

celeritas_setup_tests(SERIAL PREFIX orange
  LINK_LIBRARIES testcel_orange Celeritas::orange
)

#-----------------------------------------------------------------------------#
# Base
celeritas_add_test(orange/BoundingBox.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(orange/BoundingBoxUtils.test.cc)
celeritas_add_test(orange/MatrixUtils.test.cc)
celeritas_add_test(orange/OrangeTypes.test.cc)
celeritas_add_device_test(orange/Orange)

# Base detail
celeritas_add_test(orange/detail/BIHBuilder.test.cc)
celeritas_add_test(orange/detail/BIHTraverser.test.cc)
celeritas_add_test(orange/detail/BIHUtils.test.cc)
celeritas_add_test(orange/detail/UniverseIndexer.test.cc)

#-----------------------------------------------------------------------------#
# Construct
set(CELERITASTEST_PREFIX orange/construct)
celeritas_add_test(orange/construct/CsgTree.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(orange/construct/CsgTreeUtils.test.cc)
celeritas_add_test(orange/construct/LocalSurfaceInserter.test.cc)

#-------------------------------------#
# Transforms
set(CELERITASTEST_PREFIX orange/transform)
celeritas_add_test(orange/transform/Transformation.test.cc)
celeritas_add_test(orange/transform/Translation.test.cc)
celeritas_add_test(orange/transform/VariantTransform.test.cc)

#-------------------------------------#
# Surfaces
set(CELERITASTEST_PREFIX orange/surf)
celeritas_add_test(orange/surf/ConeAligned.test.cc)
celeritas_add_test(orange/surf/CylAligned.test.cc)
celeritas_add_test(orange/surf/CylCentered.test.cc)
celeritas_add_test(orange/surf/GeneralQuadric.test.cc)
celeritas_add_test(orange/surf/Plane.test.cc)
celeritas_add_test(orange/surf/PlaneAligned.test.cc)
celeritas_add_test(orange/surf/SimpleQuadric.test.cc)
celeritas_add_test(orange/surf/Sphere.test.cc)
celeritas_add_test(orange/surf/SphereCentered.test.cc)

celeritas_add_test(orange/surf/FaceNamer.test.cc)
celeritas_add_test(orange/surf/RecursiveSimplifier.test.cc)
celeritas_add_test(orange/surf/SoftSurfaceEqual.test.cc)
celeritas_add_test(orange/surf/SurfaceClipper.test.cc)
celeritas_add_test(orange/surf/SurfaceSimplifier.test.cc)
celeritas_add_device_test(orange/surf/LocalSurfaceVisitor)

celeritas_add_test(orange/surf/detail/QuadraticSolver.test.cc)
celeritas_add_test(orange/surf/detail/SurfaceTranslator.test.cc)
celeritas_add_test(orange/surf/detail/SurfaceTransformer.test.cc)

#-------------------------------------#
# Universe details
set(CELERITASTEST_PREFIX orange/univ)
celeritas_add_test(orange/univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(orange/univ/detail/LogicStack.test.cc)
celeritas_add_test(orange/univ/detail/RaggedRightIndexer.test.cc)
celeritas_add_test(orange/univ/detail/SurfaceFunctors.test.cc)
celeritas_add_test(orange/univ/detail/SenseCalculator.test.cc)
celeritas_add_test(orange/univ/VolumeView.test.cc)
celeritas_add_test(orange/univ/RectArrayTracker.test.cc ${_needs_json})
celeritas_add_device_test(orange/univ/SimpleUnitTracker)
celeritas_add_test(orange/univ/TrackerVisitor.test.cc ${_needs_json})

#-----------------------------------------------------------------------------#
# CELERITAS TESTS
#-----------------------------------------------------------------------------#

celeritas_add_object_library(testcel_celeritas_core
  celeritas/GeantTestBase.cc
  celeritas/GlobalGeoTestBase.cc
  celeritas/GlobalTestBase.cc
  celeritas/ImportedDataTestBase.cc
  celeritas/LazyGeoManager.cc
  celeritas/MockTestBase.cc
  celeritas/RootTestBase.cc
  celeritas/SimpleTestBase.cc
  celeritas/field/FieldTestBase.cc
  celeritas/global/AlongStepTestBase.cc
  celeritas/global/StepperTestBase.cc
  celeritas/grid/CalculatorTestBase.cc
  celeritas/io/EventIOTestBase.cc
  celeritas/phys/InteractionIO.cc
  celeritas/phys/InteractorHostTestBase.cc
  celeritas/phys/MockModel.cc
  celeritas/phys/MockProcess.cc
  celeritas/phys/SecondaryIO.cc
  celeritas/user/CaloTestBase.cc
  celeritas/user/DiagnosticTestBase.cc
  celeritas/user/ExampleMctruth.cc
  celeritas/user/MctruthTestBase.cc
  celeritas/user/StepCollectorTestBase.cc
)
set(_common_testcel_libs
  Celeritas::testcel_core Celeritas::testcel_harness Celeritas::celeritas
)
celeritas_target_link_libraries(testcel_celeritas_core
  PRIVATE ${_common_testcel_libs} ${nlohmann_json_LIBRARIES}
)

# Geometry
celeritas_add_object_library(testcel_celeritas_geo
  celeritas/GenericGeoTestBase.cc
)
target_link_libraries(testcel_celeritas_geo
  PRIVATE Celeritas::testcel_harness Celeritas::celeritas ${_geo_libs}
)

celeritas_add_library(testcel_celeritas
  $<TARGET_OBJECTS:testcel_celeritas_core>
  $<TARGET_OBJECTS:testcel_celeritas_geo>
)
celeritas_target_link_libraries(testcel_celeritas
  PRIVATE ${_common_testcel_libs} ${nlohmann_json_LIBRARIES} ${_geo_libs}
)

celeritas_setup_tests(SERIAL PREFIX celeritas
  LINK_LIBRARIES ${_common_testcel_libs} Celeritas::testcel_celeritas
)

#-----------------------------------------------------------------------------#
# (Root level tests)
celeritas_add_test(celeritas/Constants.test.cc
  LINK_LIBRARIES ${Geant4_LIBRARIES})

#-------------------------------------#
# EM
set(CELERITASTEST_PREFIX celeritas/em)
celeritas_add_test(celeritas/em/Fluctuation.test.cc ${_fixme_single})
celeritas_add_test(celeritas/em/TsaiUrbanDistribution.test.cc ${_needs_double})

# Models
celeritas_add_test(celeritas/em/BetheHeitler.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/CombinedBrem.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/EPlusGG.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/KleinNishina.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/LivermorePE.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/MollerBhabha.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/MuBremsstrahlung.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/Rayleigh.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/RelativisticBrem.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/SeltzerBerger.test.cc ${_needs_double})
celeritas_add_test(celeritas/em/Wentzel.test.cc ${_fixme_single})

celeritas_add_test(celeritas/em/UrbanMsc.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-------------------------------------#
# External
set(CELERITASTEST_PREFIX celeritas/ext)

if(CELERITAS_USE_VecGeom)
  # Vecgeom uses global counters that interfere with copying data to GPU and
  # change the internal numbering.
  set(_vecgeom_tests
    "SimpleCmsTest.*"
  )
  if(NOT VecGeom_SURF_FOUND)
    # TODO: vecgeom surface doesn't support some of these shapes
    list(APPEND _vecgeom_tests
      "FourLevelsTest.*"
      "SolidsTest.*"
      "CmseTest.*"
    )
  endif()
  if(CELERITAS_USE_Geant4)
    list(APPEND _vecgeom_tests
      "FourLevelsGeantTest.*"
      "SolidsGeantTest.*"
    )
  endif()
  celeritas_add_device_test(celeritas/ext/Vecgeom
    FILTER
      ${_vecgeom_tests}
  )
endif()
if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  celeritas_add_test(celeritas/ext/GeantGeo.test.cc
    LINK_LIBRARIES ${Geant4_LIBRARIES}
  )
  celeritas_add_test(celeritas/ext/GeantGeoUtils.test.cc
    LINK_LIBRARIES ${Geant4_LIBRARIES}
  )
endif()

celeritas_add_test(celeritas/ext/GeantImporter.test.cc
  ${_needs_geant4} LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
  FILTER
    "FourSteelSlabs*"
    "TestEm3*"
    "OneSteelSphere.*"
    "OneSteelSphereGG.*"
    "Solids.*"
  )
celeritas_add_test(celeritas/ext/GeantVolumeMapper.test.cc ${_needs_geant4}
  LINK_LIBRARIES ${Geant4_LIBRARIES} Celeritas::orange)
celeritas_add_test(celeritas/ext/RootImporter.test.cc ${_needs_root})

#-------------------------------------#
# Field
set(CELERITASTEST_PREFIX celeritas/field)

celeritas_add_test(celeritas/field/Fields.test.cc)
celeritas_add_test(celeritas/field/Steppers.test.cc)
celeritas_add_test(celeritas/field/FieldDriver.test.cc)
celeritas_add_test(celeritas/field/FieldPropagator.test.cc
  ${_needs_geo})
celeritas_add_test(celeritas/field/LinearPropagator.test.cc
  ${_needs_geo} LINK_LIBRARIES ${_geo_libs})
celeritas_add_test(celeritas/field/MagFieldEquation.test.cc)

#-------------------------------------#
# Geo
set(CELERITASTEST_PREFIX celeritas/geo)

set(_geo_args GPU ${_needs_geo} ${_needs_double}
  SOURCES celeritas/geo/HeuristicGeoTestBase.cc)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _geo_args celeritas/geo/HeuristicGeoTestBase.cu)
endif()
celeritas_add_test(celeritas/geo/Geometry.test.cc
  ${_geo_args})

if(NOT (CELERITAS_USE_Geant4 OR CELERITAS_USE_ROOT))
  set(_needs_geant_or_root DISABLE)
else()
  set(_needs_geant_or_root)
endif()
celeritas_add_test(celeritas/geo/GeoMaterial.test.cc
  ${_needs_geant_or_root} ${_needs_geo} ${_optional_geant4_env})

#-------------------------------------#
# Global
set(CELERITASTEST_PREFIX celeritas/global)
celeritas_add_test(celeritas/global/ActionRegistry.test.cc)

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set(_along_step_filter
    FILTER
      "-Em3*:SimpleCms*"
      "Em3AlongStepTest.nofluct_nomsc"
      "Em3AlongStepTest.msc_nofluct"
      "Em3AlongStepTest.msc_nofluct_finegrid"
      "Em3AlongStepTest.fluct_nomsc"
      "SimpleCmsAlongStepTest.msc_field"
      "SimpleCmsAlongStepTest.msc_field_finegrid"
      "SimpleCmsRZFieldAlongStepTest.msc_rzfield"
      "SimpleCmsRZFieldAlongStepTest.msc_rzfield_finegrid"
  )
  set(_stepper_filter
    FILTER
      "-TestEm*:OneSteelSphere*"
      "TestEm3Compton.*"
      "TestEm3NoMsc.*"
      "TestEm3Msc.*"
      "TestEm3MscNofluct.*"
      "TestEm15FieldMsc.*"
      "OneSteelSphere.*"
  )
elseif(CELERITAS_USE_Geant4)
set(_along_step_filter
    FILTER
      "-Em3*:SimpleCms*"
      "Em3AlongStepTest.nofluct_nomsc"
      "Em3AlongStepTest.fluct_nomsc"
  )
  set(_stepper_filter
    FILTER
      "-TestEm*:OneSteelSphere*"
      "TestEm3Compton.*"
      "TestEm3Msc.*"
      "TestEm3MscNofluct.*"
      "TestEm15FieldMsc.*"
    )
  else()
  set(_along_step_filter)
  set(_stepper_filter)
endif()

celeritas_add_test(celeritas/global/AlongStep.test.cc
  NT 1 ${_needs_geo} ${_optional_geant4_env}
  ${_along_step_filter}
)
celeritas_add_test(celeritas/global/KernelContextException.test.cc
  NT 1 ${_needs_geo} LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
)
celeritas_add_test(celeritas/global/Stepper.test.cc
  GPU NT 4 ${_needs_geo} ${_optional_geant4_env}
  ${_stepper_filter}
)

#-------------------------------------#
# Grid
set(CELERITASTEST_PREFIX celeritas/grid)
celeritas_add_test(celeritas/grid/GenericXsCalculator.test.cc)
celeritas_add_test(celeritas/grid/GridIdFinder.test.cc)
celeritas_add_test(celeritas/grid/InverseRangeCalculator.test.cc)
celeritas_add_test(celeritas/grid/PolyEvaluator.test.cc)
celeritas_add_test(celeritas/grid/RangeCalculator.test.cc)
celeritas_add_test(celeritas/grid/ValueGridBuilder.test.cc)
celeritas_add_test(celeritas/grid/ValueGridInserter.test.cc)
celeritas_add_test(celeritas/grid/VectorUtils.test.cc)
celeritas_add_test(celeritas/grid/XsCalculator.test.cc)

#-------------------------------------#
# IO
set(CELERITASTEST_PREFIX celeritas/io)
celeritas_add_test(celeritas/io/EventIO.test.cc ${_needs_hepmc}
  LINK_LIBRARIES ${HepMC3_LIBRARIES})
celeritas_add_test(celeritas/io/RootEventIO.test.cc ${_needs_root})
celeritas_add_test(celeritas/io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-------------------------------------#
# Mat
set(CELERITASTEST_PREFIX celeritas/mat)
celeritas_add_test(celeritas/mat/IsotopeSelector.test.cc)
celeritas_add_test(celeritas/mat/ElementSelector.test.cc)
celeritas_add_device_test(celeritas/mat/Material)

#-------------------------------------#
# Phys
set(CELERITASTEST_PREFIX celeritas/phys)
celeritas_add_test(celeritas/phys/CutoffParams.test.cc)
celeritas_add_device_test(celeritas/phys/Particle)
celeritas_add_device_test(celeritas/phys/Physics)
celeritas_add_test(celeritas/phys/PhysicsStepUtils.test.cc)
celeritas_add_test(celeritas/phys/PrimaryGenerator.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(celeritas/phys/ProcessBuilder.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Random
set(CELERITASTEST_PREFIX celeritas/random)

celeritas_add_device_test(celeritas/random/RngEngine)
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
celeritas_add_test(celeritas/track/Sim.test.cc ${_needs_geant4})
celeritas_add_test(celeritas/track/TrackSort.test.cc GPU ${_needs_geant4} ${_needs_geo})

set(_trackinit_sources
  celeritas/track/MockInteractAction.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _trackinit_sources
    celeritas/track/MockInteractAction.cu
  )
endif()

celeritas_add_test(celeritas/track/TrackInit.test.cc
  GPU ${_needs_geo} SOURCES ${_trackinit_sources}
)

#-------------------------------------#
# User
set(CELERITASTEST_PREFIX celeritas/user)

if(CELERITAS_USE_Geant4)
  set(_diagnostic_filter
    FILTER
      "-TestEm3*"
      "TestEm3*"
  )
else()
  set(_diagnostic_filter)
endif()
if(CELERITAS_CORE_GEO STREQUAL "Geant4")
  # EM track gets stuck
  set(_fails_g4geo DISABLE)
endif()

celeritas_add_test(celeritas/user/DetectorSteps.test.cc GPU)
celeritas_add_test(celeritas/user/Diagnostic.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_needs_geo} ${_fails_g4geo} ${_needs_double}
  ${_diagnostic_filter}
)
celeritas_add_test(celeritas/user/StepCollector.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_needs_geo} ${_fixme_single}
)

#-----------------------------------------------------------------------------#
# ACCELERITAS TESTS
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4)
  celeritas_add_library(testcel_accel
    accel/SDTestBase.cc
    accel/SimpleSensitiveDetector.cc
  )
  celeritas_target_link_libraries(testcel_accel
    PUBLIC testcel_celeritas testcel_core testcel_harness Celeritas::accel
  )
  celeritas_setup_tests(SERIAL PREFIX accel
    LINK_LIBRARIES testcel_accel Celeritas::accel Celeritas::celeritas
  )

  celeritas_add_test(accel/ExceptionConverter.test.cc)
  celeritas_add_test(accel/HepMC3PrimaryGenerator.test.cc
    ENVIRONMENT "${_geant4_test_env}")
  celeritas_add_test(accel/RZMapMagneticField.test.cc)
  celeritas_add_test(accel/detail/HitManager.test.cc
    ENVIRONMENT "${_geant4_test_env}")
  celeritas_add_test(accel/detail/HitProcessor.test.cc
    ENVIRONMENT "${_geant4_test_env}")
  if(CELERITAS_REAL_TYPE STREQUAL "double")
    # This test requires Geant4 *geometry* which is incompatible
    # with single-precision
    celeritas_add_test(accel/detail/TouchableUpdater.test.cc)
  endif()
endif()

#-----------------------------------------------------------------------------#
# DATA UPDATE
#-----------------------------------------------------------------------------#

function(celeritas_add_rootupdate_target target filename)
  set(_basename "${CMAKE_CURRENT_SOURCE_DIR}/celeritas/data/${filename}")
  if(CELERITAS_USE_Geant4)
    set(_cmake_env
      "${CMAKE_COMMAND}" "-E" "env" "${_geant4_test_env}"
    )
  else()
    set(_cmake_env)
  endif()
  add_custom_target(${target}
    COMMAND
      ${_cmake_env}
      "$<TARGET_FILE:celer-export-geant>"
      "${_basename}.gdml"
      "${_basename}.geant.json"
      "${_basename}.root"
    VERBATIM
  )
  add_dependencies(${target} celer-export-geant)
endfunction()

celeritas_add_rootupdate_target(update-four-slabs four-steel-slabs)
celeritas_add_rootupdate_target(update-simple-cms simple-cms)

#-----------------------------------------------------------------------------#
