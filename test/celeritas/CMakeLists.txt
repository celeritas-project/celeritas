#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

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

if(CELERITAS_DEBUG)
  set(_disable_if_debug DISABLE)
endif()

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
else()
  set(_optional_geant4_env ENVIRONMENT "${CELERITASTEST_G4ENV}")
  set(_needs_geant4 ${_optional_geant4_env})
endif()

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()

if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE" AND NOT CELERITAS_USE_JSON)
  set(_needs_geo DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  set(_core_geo_libs Celeritas::testcel_orange Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  set(_core_geo_libs Celeritas::testcel_geocel ${VecGeom_LIBRARIES})
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  set(_core_geo_libs Celeritas::testcel_geocel ${Geant4_LIBRARIES})
endif()

set(_all_geo_libs
  # Geometry includes
  Celeritas::orange ${VecGeom_LIBRARIES} ${Geant4_LIBRARIES}
  # Test harness instantiations
  Celeritas::testcel_geocel Celeritas::testcel_orange
)

if(CELERITAS_UNITS STREQUAL "CGS")
  set(_fixme_cgs)
else()
  # Unit tests haven't yet been updated to include the correct units
  set(_fixme_cgs DISABLE)
endif()

#-----------------------------------------------------------------------------#
# LIBRARY
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_celeritas
  GeantTestBase.cc
  GlobalGeoTestBase.cc
  GlobalTestBase.cc
  ImportedDataTestBase.cc
  MockTestBase.cc
  RootTestBase.cc
  SimpleTestBase.cc
  field/FieldTestBase.cc
  global/AlongStepTestBase.cc
  global/StepperTestBase.cc
  grid/CalculatorTestBase.cc
  io/EventIOTestBase.cc
  neutron/NeutronTestBase.cc
  optical/OpticalTestBase.cc
  phys/InteractionIO.cc
  phys/InteractorHostTestBase.cc
  phys/MockModel.cc
  phys/MockProcess.cc
  phys/SecondaryIO.cc
  user/CaloTestBase.cc
  user/DiagnosticTestBase.cc
  user/ExampleMctruth.cc
  user/MctruthTestBase.cc
  user/StepCollectorTestBase.cc
)

celeritas_target_link_libraries(testcel_celeritas
  PUBLIC
    Celeritas::testcel_harness Celeritas::celeritas
  PRIVATE
    Celeritas::testcel_core Celeritas::testcel_orange Celeritas::testcel_geocel
    ${nlohmann_json_LIBRARIES}
)

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL PREFIX celeritas
  LINK_LIBRARIES Celeritas::testcel_celeritas Celeritas::testcel_core
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(Constants.test.cc
  LINK_LIBRARIES ${Geant4_LIBRARIES})
celeritas_add_test(Units.test.cc)

#-----------------------------------------------------------------------------#
# EM
set(CELERITASTEST_PREFIX celeritas/em)
celeritas_add_test(em/Fluctuation.test.cc ${_fixme_single})
celeritas_add_test(em/TsaiUrbanDistribution.test.cc ${_needs_double})

# Models
celeritas_add_test(em/BetheHeitler.test.cc ${_needs_double})
celeritas_add_test(em/CombinedBrem.test.cc ${_needs_double})
celeritas_add_test(em/EPlusGG.test.cc ${_needs_double})
celeritas_add_test(em/KleinNishina.test.cc ${_needs_double})
celeritas_add_test(em/LivermorePE.test.cc ${_needs_double})
celeritas_add_test(em/MollerBhabha.test.cc ${_needs_double})
celeritas_add_test(em/MuBremsstrahlung.test.cc ${_needs_double})
celeritas_add_test(em/Rayleigh.test.cc ${_needs_double})
celeritas_add_test(em/RelativisticBrem.test.cc ${_needs_double})
celeritas_add_test(em/SeltzerBerger.test.cc ${_needs_double})
celeritas_add_test(em/CoulombScattering.test.cc ${_fixme_single})

celeritas_add_test(em/UrbanMsc.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# External
set(CELERITASTEST_PREFIX celeritas/ext)

celeritas_add_test(ext/GeantImporter.test.cc
  ${_needs_geant4}
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
  FILTER
    "FourSteelSlabs*"
    "TestEm3*"
    "OneSteelSphere.*"
    "OneSteelSphereGG.*"
    "Solids.*"
  )
celeritas_add_test(ext/GeantVolumeMapper.test.cc ${_needs_geant4}
  LINK_LIBRARIES ${Geant4_LIBRARIES} Celeritas::orange)
celeritas_add_test(ext/RootImporter.test.cc ${_needs_root})

#-----------------------------------------------------------------------------#
# Field
set(CELERITASTEST_PREFIX celeritas/field)

celeritas_add_test(field/Fields.test.cc)
celeritas_add_test(field/Steppers.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldDriver.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldPropagator.test.cc
 ${_fixme_cgs}
  LINK_LIBRARIES ${_core_geo_libs})
celeritas_add_test(field/LinearPropagator.test.cc
  ${_needs_geo}
  LINK_LIBRARIES ${_all_geo_libs})
celeritas_add_test(field/MagFieldEquation.test.cc)

#-----------------------------------------------------------------------------#
# Geo
set(CELERITASTEST_PREFIX celeritas/geo)

set(_geo_args GPU ${_needs_geo} ${_needs_double}
  SOURCES geo/HeuristicGeoTestBase.cc)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _geo_args geo/HeuristicGeoTestBase.cu)
endif()
celeritas_add_test(geo/Geometry.test.cc
  ${_geo_args} ${_fixme_cgs})

if(NOT (CELERITAS_USE_Geant4 OR CELERITAS_USE_ROOT))
  set(_needs_geant_or_root DISABLE)
else()
  set(_needs_geant_or_root)
endif()
celeritas_add_test(geo/GeoMaterial.test.cc
  ${_needs_geant_or_root} ${_needs_geo} ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Global
set(CELERITASTEST_PREFIX celeritas/global)
celeritas_add_test(global/ActionRegistry.test.cc)

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set(_along_step_filter
    FILTER
      "-Em3*:SimpleCms*:LeadBox*"
      "Em3AlongStepTest.nofluct_nomsc"
      "Em3AlongStepTest.msc_nofluct"
      "Em3AlongStepTest.msc_nofluct_finegrid"
      "Em3AlongStepTest.fluct_nomsc"
      "SimpleCmsAlongStepTest.msc_field"
      "SimpleCmsAlongStepTest.msc_field_finegrid"
      "SimpleCmsRZFieldAlongStepTest.msc_rzfield"
      "SimpleCmsRZFieldAlongStepTest.msc_rzfield_finegrid"
      "LeadBox*"
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
      "-Em3*:SimpleCms*:LeadBox*"
      "Em3AlongStepTest.nofluct_nomsc"
      "Em3AlongStepTest.fluct_nomsc"
      "LeadBox*"
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

celeritas_add_test(global/AlongStep.test.cc
  NT 1 ${_needs_geo} ${_optional_geant4_env}
  ${_along_step_filter}
)
celeritas_add_test(global/KernelContextException.test.cc
  NT 1 ${_needs_geo} LINK_LIBRARIES ${nlohmann_json_LIBRARIES}
)
celeritas_add_test(global/Stepper.test.cc
  GPU NT 4 ${_needs_geo} ${_optional_geant4_env}
  ${_stepper_filter}
)

#-----------------------------------------------------------------------------#
# Grid
set(CELERITASTEST_PREFIX celeritas/grid)
celeritas_add_test(grid/GenericCalculator.test.cc)
celeritas_add_test(grid/GridIdFinder.test.cc)
celeritas_add_test(grid/InverseRangeCalculator.test.cc)
celeritas_add_test(grid/PolyEvaluator.test.cc)
celeritas_add_test(grid/RangeCalculator.test.cc)
celeritas_add_test(grid/ValueGridBuilder.test.cc)
celeritas_add_test(grid/ValueGridInserter.test.cc)
celeritas_add_test(grid/VectorUtils.test.cc)
celeritas_add_test(grid/XsCalculator.test.cc)

#-----------------------------------------------------------------------------#
# IO
set(CELERITASTEST_PREFIX celeritas/io)
celeritas_add_test(io/EventIO.test.cc ${_needs_hepmc}
  LINK_LIBRARIES ${HepMC3_LIBRARIES})
celeritas_add_test(io/ImportUnits.test.cc)
celeritas_add_test(io/RootEventIO.test.cc ${_needs_root})
celeritas_add_test(io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-----------------------------------------------------------------------------#
# Optical
set(CELERITASTEST_PREFIX celeritas/optical)
celeritas_add_test(optical/Cerenkov.test.cc)
celeritas_add_test(optical/ScintillationGenerator.test.cc)

#-----------------------------------------------------------------------------#
# Mat
set(CELERITASTEST_PREFIX celeritas/mat)
celeritas_add_test(mat/IsotopeSelector.test.cc)
celeritas_add_test(mat/ElementSelector.test.cc)
celeritas_add_device_test(mat/Material)

#-----------------------------------------------------------------------------#
# Neutron
set(CELERITASTEST_PREFIX celeritas/neutron)
celeritas_add_test(neutron/NeutronElastic.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# Phys
set(CELERITASTEST_PREFIX celeritas/phys)
celeritas_add_test(phys/CutoffParams.test.cc)
celeritas_add_device_test(phys/Particle)
celeritas_add_device_test(phys/Physics)
celeritas_add_test(phys/PhysicsStepUtils.test.cc)
celeritas_add_test(phys/PrimaryGenerator.test.cc
  LINK_LIBRARIES ${nlohmann_json_LIBRARIES})
celeritas_add_test(phys/ProcessBuilder.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Random
set(CELERITASTEST_PREFIX celeritas/random)

celeritas_add_device_test(random/RngEngine)
celeritas_add_test(random/Selector.test.cc)
celeritas_add_test(random/RngReseed.test.cc)
celeritas_add_test(random/XorwowRngEngine.test.cc GPU)

celeritas_add_test(random/distribution/BernoulliDistribution.test.cc)
celeritas_add_test(random/distribution/ExponentialDistribution.test.cc)
celeritas_add_test(random/distribution/GammaDistribution.test.cc)
celeritas_add_test(random/distribution/IsotropicDistribution.test.cc)
celeritas_add_test(random/distribution/NormalDistribution.test.cc)
celeritas_add_test(random/distribution/PoissonDistribution.test.cc)
celeritas_add_test(random/distribution/RadialDistribution.test.cc)
celeritas_add_test(random/distribution/ReciprocalDistribution.test.cc)
celeritas_add_test(random/distribution/UniformBoxDistribution.test.cc)
celeritas_add_test(random/distribution/UniformRealDistribution.test.cc)

if(CELERITAS_USE_CUDA)
  celeritas_add_test(random/curand/CurandPerformance.test.cc GPU
    SOURCES random/curand/CurandPerformance.test.cu
    LINK_LIBRARIES Celeritas::DeviceToolkit
    ${_disable_if_debug}
  )
endif()

#-----------------------------------------------------------------------------#
# Track
set(CELERITASTEST_PREFIX celeritas/track)
celeritas_add_test(track/Sim.test.cc ${_needs_geant4})
celeritas_add_test(track/TrackSort.test.cc GPU ${_needs_geant4} ${_needs_geo})

set(_trackinit_sources
  track/MockInteractAction.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _trackinit_sources
    track/MockInteractAction.cu
  )
endif()

celeritas_add_test(track/TrackInit.test.cc
  GPU ${_needs_geo} SOURCES ${_trackinit_sources}
)

#-----------------------------------------------------------------------------#
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

celeritas_add_test(user/DetectorSteps.test.cc GPU)
celeritas_add_test(user/Diagnostic.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_needs_geo} ${_fails_g4geo} ${_needs_double}
  ${_diagnostic_filter}
)
celeritas_add_test(user/StepCollector.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_needs_geo} ${_fixme_single}
)

#-----------------------------------------------------------------------------#
# DATA UPDATE
#-----------------------------------------------------------------------------#

function(celeritas_add_rootupdate_target target filename)
  set(_gdml "${PROJECT_SOURCE_DIR}/test/geocel/data/${filename}.gdml")
  set(_basename "${CMAKE_CURRENT_SOURCE_DIR}/data/${filename}")
  if(CELERITAS_USE_Geant4)
    set(_cmake_env
      "${CMAKE_COMMAND}" "-E" "env" "${CELERITASTEST_G4ENV}"
    )
  else()
    set(_cmake_env)
  endif()
  add_custom_target(${target}
    COMMAND
      ${_cmake_env}
      "$<TARGET_FILE:celer-export-geant>"
      "${_gdml}"
      "${_basename}.geant.json"
      "${_basename}.root"
    VERBATIM
  )
  add_dependencies(${target} celer-export-geant)
endfunction()

celeritas_add_rootupdate_target(update-four-slabs four-steel-slabs)
celeritas_add_rootupdate_target(update-simple-cms simple-cms)

#-----------------------------------------------------------------------------#
