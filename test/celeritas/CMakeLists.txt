#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
  set(_g4_geo_libs)
else()
  set(_optional_geant4_env ENVIRONMENT "${CELER_G4ENV}")
  set(_needs_geant4 ${_optional_geant4_env})
  celeritas_get_g4libs(_g4_geo_libs geometry)
endif()

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()

if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  set(_core_geo_libs testcel_orange Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  set(_core_geo_libs testcel_geocel ${VecGeom_LIBRARIES})
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  set(_core_geo_libs testcel_geocel ${Geant4_LIBRARIES})
endif()

set(_all_geo_libs
  # Geometry includes
  Celeritas::orange ${VecGeom_LIBRARIES} ${Geant4_LIBRARIES}
  # Test harness instantiations
  testcel_geocel testcel_orange
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

celeritas_add_test_library(testcel_celeritas
  GeantTestBase.cc
  GlobalGeoTestBase.cc
  GlobalTestBase.cc
  ImportedDataTestBase.cc
  InvalidOrangeTestBase.cc
  MockTestBase.cc
  RootTestBase.cc
  SimpleTestBase.cc
  em/MscTestBase.cc
  field/FieldTestBase.cc
  global/AlongStepTestBase.cc
  global/DummyAction.cc
  global/StepperTestBase.cc
  grid/CalculatorTestBase.cc
  io/EventIOTestBase.cc
  neutron/NeutronTestBase.cc
  optical/OpticalMockTestBase.cc
  optical/OpticalTestBase.cc
  optical/InteractorHostTestBase.cc
  optical/ValidationUtils.cc
  phys/InteractionIO.cc
  phys/InteractorHostTestBase.cc
  phys/MockModel.cc
  phys/MockProcess.cc
  phys/SecondaryIO.cc
  user/CaloTestBase.cc
  user/DiagnosticTestBase.cc
  user/ExampleInstanceCalo.cc
  user/ExampleMctruth.cc
  user/MctruthTestBase.cc
  user/SimpleLoopTestBase.cc
)

celeritas_target_link_libraries(testcel_celeritas
  PUBLIC
    testcel_harness Celeritas::celeritas
  PRIVATE
    testcel_core testcel_orange testcel_geocel
    nlohmann_json::nlohmann_json ${_g4_geo_libs}
)

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_celeritas testcel_core
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(Constants.test.cc
  LINK_LIBRARIES ${Geant4_LIBRARIES})
celeritas_add_test(Units.test.cc)

#-----------------------------------------------------------------------------#
# Decay

# Muon
celeritas_add_test(decay/MuDecay.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# EM

# Distributions
celeritas_add_test(em/distribution/EnergyLossHelper.test.cc ${_fixme_single})
celeritas_add_test(em/distribution/MuAngularDistribution.test.cc ${_needs_double})
celeritas_add_test(em/distribution/TsaiUrbanDistribution.test.cc ${_needs_double})

# Cross section calculation
celeritas_add_test(em/xs/NuclearFormFactors.test.cc)

# Models
celeritas_add_test(em/BetheHeitler.test.cc ${_needs_double})
celeritas_add_test(em/BetheBloch.test.cc ${_needs_double})
celeritas_add_test(em/BraggICRU73QO.test.cc ${_needs_double})
celeritas_add_test(em/CombinedBrem.test.cc ${_needs_double})
celeritas_add_test(em/EPlusGG.test.cc ${_needs_double})
celeritas_add_test(em/KleinNishina.test.cc ${_needs_double})
celeritas_add_test(em/LivermorePE.test.cc ${_needs_double})
celeritas_add_test(em/MollerBhabha.test.cc ${_needs_double})
celeritas_add_test(em/MuBetheBloch.test.cc ${_needs_double})
celeritas_add_test(em/MuBremsstrahlung.test.cc ${_needs_double})
celeritas_add_test(em/MuPairProduction.test.cc ${_needs_double} ${_needs_root})
celeritas_add_test(em/Rayleigh.test.cc ${_needs_double})
celeritas_add_test(em/RelativisticBrem.test.cc ${_needs_double})
celeritas_add_test(em/SeltzerBerger.test.cc ${_needs_double})
celeritas_add_test(em/CoulombScattering.test.cc ${_fixme_single})

celeritas_add_test(em/UrbanMsc.test.cc ${_needs_root}
  ${_optional_geant4_env})
celeritas_add_test(em/WentzelVIMsc.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# External
set(_import_filter
  "FourSteelSlabs*"
  "TestEm3*"
  "OneSteelSphere.*"
  "OneSteelSphereGG.*"
  "LarSphere.*"
  "LarSphereExtramat.*"
  "Solids.*"
)
if(Geant4_VERSION VERSION_LESS 11.0)
  list(REMOVE_ITEM _import_filter
    "OneSteelSphere.*" # cutoffs changed
  )
endif()

celeritas_add_test(ext/GeantImporter.test.cc
  ${_needs_geant4}
  LINK_LIBRARIES nlohmann_json::nlohmann_json
  FILTER ${_import_filter}
)
celeritas_add_test(ext/GeantVolumeMapper.test.cc ${_needs_geant4}
  LINK_LIBRARIES ${Geant4_LIBRARIES} Celeritas::orange)
celeritas_add_test(ext/RootImporter.test.cc ${_needs_root})
celeritas_add_test(ext/RootJsonDumper.test.cc ${_needs_root})

#-----------------------------------------------------------------------------#
# Field

celeritas_add_test(field/Fields.test.cc)
celeritas_add_test(field/Steppers.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldDriver.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldPropagator.test.cc
 ${_fixme_cgs}
  LINK_LIBRARIES ${_core_geo_libs})
celeritas_add_test(field/LinearPropagator.test.cc
  LINK_LIBRARIES ${_all_geo_libs})
celeritas_add_test(field/MagFieldEquation.test.cc)

#-----------------------------------------------------------------------------#
# Geo

set(_geo_args GPU ${_needs_double} ${_fixme_cgs}
  LINK_LIBRARIES ${_all_geo_libs}
  SOURCES geo/HeuristicGeoTestBase.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _geo_args geo/HeuristicGeoTestBase.cu)
endif()
celeritas_add_test(geo/Geometry.test.cc ${_geo_args})

if(NOT (CELERITAS_USE_Geant4 OR CELERITAS_USE_ROOT))
  set(_needs_geant_or_root DISABLE)
else()
  set(_needs_geant_or_root)
endif()
celeritas_add_test(geo/GeoMaterial.test.cc
  ${_needs_geant_or_root} ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Global
if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set(_along_step_filter
    "-Em3*:SimpleCms*:LeadBox*"
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
    "-TestEm*:LeadBox*"
    "TestEm3Compton.*"
    "TestEm3NoMsc.*"
    "TestEm3Msc.*"
    "TestEm3MscNofluct.*"
    "TestEm15FieldMsc.*"
  )
  if(Geant4_VERSION VERSION_LESS 11.0)
    list(REMOVE_ITEM _along_step_filter
      "Em3AlongStepTest.msc_nofluct_finegrid"
      "LeadBox*"
    )
  endif()
elseif(CELERITAS_USE_Geant4)
  set(_along_step_filter
    "-Em3*:SimpleCms*:LeadBox*"
    "Em3AlongStepTest.nofluct_nomsc"
    "Em3AlongStepTest.fluct_nomsc"
    "LeadBox*"
  )
  set(_stepper_filter
    "-TestEm*:LeadBox*"
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
  NT 1 ${_optional_geant4_env}
  FILTER ${_along_step_filter}
)
celeritas_add_test(global/KernelContextException.test.cc
  NT 1 LINK_LIBRARIES nlohmann_json::nlohmann_json
)
celeritas_add_test(global/Stepper.test.cc
  GPU NT 4
)
celeritas_add_test(global/StepperGeant.test.cc
  ${_needs_geant4}
  GPU NT 4
  FILTER ${_stepper_filter}
)

#-----------------------------------------------------------------------------#
# Grid
celeritas_add_test(grid/GridIdFinder.test.cc)
celeritas_add_test(grid/InverseCdfFinder.test.cc)
celeritas_add_test(grid/InverseRangeCalculator.test.cc)
celeritas_add_test(grid/NonuniformGridCalculator.test.cc)
celeritas_add_test(grid/NonuniformGridBuilder.test.cc)
celeritas_add_test(grid/NonuniformGridInserter.test.cc)
celeritas_add_test(grid/RangeCalculator.test.cc)
celeritas_add_test(grid/SplineDerivCalculator.test.cc)
celeritas_add_test(grid/SplineXsCalculator.test.cc)
celeritas_add_test(grid/ValueGridBuilder.test.cc)
celeritas_add_test(grid/ValueGridInserter.test.cc)
celeritas_add_test(grid/XsCalculator.test.cc)

#-----------------------------------------------------------------------------#
# IO
celeritas_add_test(io/EventIO.test.cc ${_needs_hepmc}
  LINK_LIBRARIES ${HepMC3_LIBRARIES})
celeritas_add_test(io/ImportUnits.test.cc)
celeritas_add_test(io/RootEventIO.test.cc ${_needs_root})
celeritas_add_test(io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-----------------------------------------------------------------------------#
# Optical
celeritas_add_test(optical/Absorption.test.cc)
celeritas_add_test(optical/Cherenkov.test.cc)
celeritas_add_test(optical/ImportedModelAdapter.test.cc)
celeritas_add_test(optical/ImportedMaterials.test.cc)
celeritas_add_test(optical/ModelImporter.test.cc)
celeritas_add_test(optical/MfpBuilder.test.cc)
celeritas_add_test(optical/OpticalCollector.test.cc ${_needs_geant4})
celeritas_add_test(optical/OpticalUtils.test.cc)
celeritas_add_test(optical/Rayleigh.test.cc)
celeritas_add_test(optical/RayleighMfpCalculator.test.cc)
celeritas_add_test(optical/OpticalRootImport.test.cc ${_needs_root})
celeritas_add_test(optical/Scintillation.test.cc)
celeritas_add_test(optical/WavelengthShift.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# Mat
celeritas_add_test(mat/IsotopeSelector.test.cc)
celeritas_add_test(mat/ElementSelector.test.cc)
celeritas_add_device_test(mat/Material)

#-----------------------------------------------------------------------------#
# Neutron
celeritas_add_test(neutron/NeutronElastic.test.cc ${_needs_double})
celeritas_add_test(neutron/NeutronInelastic.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# Phys
celeritas_add_test(phys/CutoffParams.test.cc)
celeritas_add_test(phys/FourVector.test.cc)
celeritas_add_device_test(phys/Particle)
celeritas_add_device_test(phys/Physics)
celeritas_add_test(phys/InteractionUtils.test.cc)
celeritas_add_test(phys/PhysicsStepUtils.test.cc)
celeritas_add_test(phys/PrimaryGenerator.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(phys/ProcessBuilder.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Random

celeritas_add_device_test(random/RngEngine)
celeritas_add_test(random/Selector.test.cc)
celeritas_add_test(random/RngReseed.test.cc)
celeritas_add_test(random/XorwowRngEngine.test.cc GPU)

celeritas_add_test(random/distribution/BernoulliDistribution.test.cc)
celeritas_add_test(random/distribution/ExponentialDistribution.test.cc)
celeritas_add_test(random/distribution/GammaDistribution.test.cc)
celeritas_add_test(random/distribution/InverseSquareDistribution.test.cc)
celeritas_add_test(random/distribution/IsotropicDistribution.test.cc)
celeritas_add_test(random/distribution/NormalDistribution.test.cc)
celeritas_add_test(random/distribution/PoissonDistribution.test.cc)
celeritas_add_test(random/distribution/RadialDistribution.test.cc)
celeritas_add_test(random/distribution/ReciprocalDistribution.test.cc)
celeritas_add_test(random/distribution/RejectionSampler.test.cc)
celeritas_add_test(random/distribution/UniformBoxDistribution.test.cc)
celeritas_add_test(random/distribution/UniformRealDistribution.test.cc)

#-----------------------------------------------------------------------------#
# Track
celeritas_add_test(track/Sim.test.cc ${_needs_geant4})
celeritas_add_test(track/StatusChecker.test.cc GPU)
celeritas_add_test(track/TrackSort.test.cc GPU ${_needs_geant4})

set(_trackinit_sources
  track/MockInteractAction.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _trackinit_sources
    track/MockInteractAction.cu
  )
endif()

celeritas_add_test(track/TrackInit.test.cc
  GPU SOURCES ${_trackinit_sources}
)

#-----------------------------------------------------------------------------#
# User

if(CELERITAS_USE_Geant4)
  set(_diagnostic_filter
    "-TestEm3*"
    "TestEm3*"
  )
  set(_step_filter
    "-TestEm3*:TestMultiEm3*"
    "TestEm3*"
    "TestMultiEm3*"
  )
else()
  set(_diagnostic_filter)
  set(_step_filter)
endif()
if(CELERITAS_CORE_GEO STREQUAL "Geant4")
  # EM track gets stuck
  set(_fails_g4geo DISABLE)
endif()

celeritas_add_test(user/DetectorSteps.test.cc GPU)
celeritas_add_test(user/Diagnostic.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_fails_g4geo} ${_needs_double}
  FILTER ${_diagnostic_filter}
)
celeritas_add_test(user/SlotDiagnostic.test.cc
  GPU NT 1 ${_needs_geant4} ${_needs_double}
  LINK_LIBRARIES nlohmann_json::nlohmann_json
)
celeritas_add_test(user/StepCollector.test.cc
  GPU NT 1 ${_optional_geant4_env} ${_fixme_single}
  FILTER ${_step_filter}
)

#-----------------------------------------------------------------------------#
# DATA UPDATE
#-----------------------------------------------------------------------------#

add_custom_target(update-root-test-data
  COMMENT "Regenerate ROOT test data"
)

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_ROOT)
  foreach(_filename "four-steel-slabs" "simple-cms" "lar-sphere")
    set(_gdml "${PROJECT_SOURCE_DIR}/test/geocel/data/${_filename}.gdml")
    set(_basename "${CMAKE_CURRENT_SOURCE_DIR}/data/${_filename}")
    set(_output "${_basename}.root")
    add_custom_command(
      TARGET update-root-test-data
      POST_BUILD
      COMMAND
        "${CMAKE_COMMAND}" "-E" "env" ${CELER_G4ENV}
        "$<TARGET_FILE:celer-export-geant>"
        "${_gdml}"
        "${_basename}.geant.json"
        "${_output}"
        "--gen-test"
      VERBATIM
    )
  endforeach()
endif()

#-----------------------------------------------------------------------------#
