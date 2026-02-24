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

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
  set(_needs_geant4_11 DISABLE)
else()
  set(_optional_geant4_env ENVIRONMENT "${CELER_G4ENV}")
  set(_needs_geant4 ${_optional_geant4_env})
  if(Geant4_VERSION VERSION_GREATER_EQUAL 11)
    set(_needs_geant4_11 ${_optional_geant4_env})
  else()
    set(_needs_geant4_11 DISABLE)
  endif()
  celeritas_get_g4libs(_g4_geo_lib geometry)
  celeritas_get_g4libs(_g4_global_libs global)
endif()

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
else()
  set(_needs_hepmc LINK_LIBRARIES ${HepMC3_LIBRARIES})
endif()

if(NOT CELERITAS_USE_ROOT)
  set(_needs_root DISABLE)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  set(_core_geo_libs testcel_orange Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  set(_needs_orange DISABLE)
  set(_core_geo_libs testcel_geocel ${VecGeom_LIBRARIES})
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  set(_needs_orange DISABLE)
  set(_core_geo_libs testcel_geocel ${_g4_geo_lib})
endif()

set(_all_geo_libs
  # Geometry includes
  Celeritas::orange ${VecGeom_LIBRARIES} Celeritas::ExtGeant4Geo
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

set(_sources
  GeantTestBase.cc
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
  mucf/MucfInteractorHostTestBase.cc
  neutron/NeutronTestBase.cc
  optical/OpticalMockTestBase.cc
  optical/OpticalTestBase.cc
  optical/InteractorHostTestBase.cc
  optical/SurfacePhysicsIntegrationTestBase.cc
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
celeritas_polysource_append(_sources track/MockInteractAction)
celeritas_polysource_append(_sources geo/HeuristicGeoTestBase)
celeritas_add_test_library(testcel_celeritas ${_sources})

celeritas_target_link_libraries(testcel_celeritas
  PUBLIC
    testcel_harness Celeritas::celeritas testcel_geocel
  PRIVATE
    testcel_core testcel_orange
    nlohmann_json::nlohmann_json Celeritas::ExtGeant4Geo
)

if(CELERITAS_USE_Geant4)
  celeritas_get_g4libs(_g4_libs materials digits_hits processes)

  celeritas_add_test_library(testcel_celeritas_g4
    ext/SensDetTestBase.cc
    ext/SimpleSensitiveDetector.cc
  )
  celeritas_target_link_libraries(testcel_celeritas_g4
    PUBLIC
      testcel_celeritas testcel_harness Celeritas::celeritas
      ${_g4_libs} Celeritas::ExtGeant4Geo
  )
endif()

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_celeritas testcel_core
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(Constants.test.cc LINK_LIBRARIES ${_g4_global_libs})
celeritas_add_test(Units.test.cc LINK_LIBRARIES ${_g4_global_libs})

#-----------------------------------------------------------------------------#
# Decay

# Muon
celeritas_add_test(decay/MuDecay.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# MuCF

celeritas_add_test(mucf/DDMucfInteractor.test.cc)
celeritas_add_test(mucf/DTMucfInteractor.test.cc)
celeritas_add_test(mucf/DTMixMucfModel.test.cc)
celeritas_add_test(mucf/MuonicAtomSelector.test.cc)

#-----------------------------------------------------------------------------#
# EM

# Distributions
celeritas_add_test(em/distribution/EnergyLossHelper.test.cc ${_fixme_single})
celeritas_add_test(em/distribution/MuAngularDistribution.test.cc ${_needs_double})
celeritas_add_test(em/distribution/TsaiUrbanDistribution.test.cc ${_needs_double})

# Cross section calculation
celeritas_add_test(em/xs/NuclearFormFactors.test.cc)
celeritas_add_test(em/xs/ScreeningFunctions.test.cc)

# Models
celeritas_add_test(em/BetheHeitler.test.cc ${_needs_double})
celeritas_add_test(em/BetheBloch.test.cc ${_needs_double})
celeritas_add_test(em/BraggICRU73QO.test.cc ${_needs_double})
celeritas_add_test(em/ElectroNuclear.test.cc ${_needs_double}
  ${_needs_geant4_11})
celeritas_add_test(em/EPlusGG.test.cc ${_needs_double})
celeritas_add_test(em/GammaNuclear.test.cc ${_needs_double} ${_needs_geant4_11})
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

celeritas_add_test(em/UrbanMsc.test.cc ${_optional_geant4_env})
celeritas_add_test(em/WentzelVIMsc.test.cc ${_needs_root}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
if(CELERITAS_USE_Geant4)
  celeritas_add_test(g4/MagneticField.test.cc
    LINK_LIBRARIES ${_g4_geo_lib})
endif()

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

celeritas_add_test(ext/JsonIO.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)

celeritas_add_test(ext/RootImporter.test.cc ${_needs_root})
celeritas_add_test(ext/RootJsonDumper.test.cc ${_needs_root})

if(CELERITAS_USE_Geant4)
  celeritas_add_test(ext/GeantSd.test.cc
    LINK_LIBRARIES testcel_celeritas_g4
    ENVIRONMENT "${CELER_G4ENV}")
  celeritas_add_test(ext/detail/HitProcessor.test.cc
    LINK_LIBRARIES testcel_celeritas_g4
    ENVIRONMENT "${CELER_G4ENV}")
  celeritas_add_test(ext/GeantTrackReconstruction.test.cc
    LINK_LIBRARIES testcel_celeritas_g4
    ENVIRONMENT "${CELER_G4ENV}")

  celeritas_add_test(ext/detail/LevelTouchableUpdater.test.cc
    LINK_LIBRARIES testcel_celeritas_g4)
  if(CELERITAS_REAL_TYPE STREQUAL "double")
    # This test requires Geant4 *test harness* which is double-precision only,
    # incompatible with single-precision build (GeantGeoTestBase is not built in
    # test/geocel/CMakeLists.txt unless using double precision)
    celeritas_add_test(ext/detail/NaviTouchableUpdater.test.cc
      LINK_LIBRARIES testcel_celeritas_g4 testcel_geocel)
  endif()
endif()
#-----------------------------------------------------------------------------#
# Field

if(CELERITAS_CORE_GEO STREQUAL "VecGeom"
 AND (CELERITAS_USE_CUDA OR CELERITAS_VecGeom_SURFACE))
  # VecGeom surface cannot rebuild the geometry
  set(_fieldprop_filter
    FILTER
    "-Layers*:SimpleCms*:Cmse*"
    "Layers*"
    "SimpleCms*"
    "Cmse*"
  )
endif()

celeritas_add_device_test(field/Fields
  LINK_LIBRARIES Celeritas::Extcovfie
)
celeritas_add_test(field/Integrators.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldSubstepper.test.cc ${_fixme_cgs})
celeritas_add_test(field/FieldPropagator.test.cc
  LINK_LIBRARIES ${_core_geo_libs}
  ${_fieldprop_filter}
  ${_fixme_cgs}
)
celeritas_add_test(field/LinearPropagator.test.cc
  LINK_LIBRARIES ${_all_geo_libs})
celeritas_add_test(field/MagFieldEquation.test.cc)

#-----------------------------------------------------------------------------#
# Geo

celeritas_add_test(geo/Geometry.test.cc  GPU ${_needs_double} ${_fixme_cgs}
  FILTER "SimpleCms*" "TestEm3*" "ThreeSpheres*")

if(NOT (CELERITAS_USE_Geant4 OR CELERITAS_USE_ROOT))
  set(_needs_geant_or_root DISABLE)
else()
  set(_needs_geant_or_root)
endif()

if(CELERITAS_USE_Geant4)
  set(_geo_mat_filter
    "-Em3*:MultiLevel*"
    "Em3Test.*"
  )
  if(NOT CELERITAS_CORE_GEO STREQUAL "ORANGE")
    # TODO: ORANGE reflection is not implemented
    list(APPEND _geo_mat_filter
      "MultiLevelTest.*"
    )
  endif()
elseif(CELERITAS_USE_ROOT)
  set(_geo_mat_filter
    "SimpleCmsTest"
  )
else()
  set(_geo_mat_filter DISABLE)
endif()

if(CELERITAS_VecGeom_SURFACE)
  set(_fails_vgsurf DISABLE)
endif()

celeritas_add_test(geo/GeoMaterial.test.cc
  FILTER ${_geo_mat_filter}
  ${_optional_geant4_env})

#-----------------------------------------------------------------------------#
# Global
if(CELERITAS_USE_Geant4)
  set(_along_step_filter
    "-Em3*:SimpleCms*:LeadBox*:Mock*"
    "Mock*"
    "Em3AlongStepTest.nofluct_nomsc"
    "Em3AlongStepTest.msc_nofluct"
    "Em3AlongStepTest.msc_nofluct_finegrid"
    "Em3AlongStepTest.fluct_nomsc"
    "SimpleCmsAlongStepTest.msc_field"
    "SimpleCmsAlongStepTest.msc_field_finegrid"
    "SimpleCmsFieldVolAlongStepTest.msc_field"
    "SimpleCmsRZFieldAlongStepTest.msc_rzfield"
    "SimpleCmsRZFieldAlongStepTest.msc_rzfield_finegrid"
    "LeadBoxAlongStepTest.*"
  )
  set(_stepper_filter
    "-TestEm*:LeadBox*"
    "TestEm3Compton.*"
    "TestEm3NoMsc.*"
    "TestEm3Msc.*"
    "TestEm3MscNofluct.*"
    "TestEm3MscNoIntegral.*"
    "TestEm15FieldMsc.*"
  )
  if(Geant4_VERSION VERSION_LESS 11.0)
    list(REMOVE_ITEM _along_step_filter
      "Em3AlongStepTest.msc_nofluct_finegrid"
      "LeadBox*"
    )
  endif()
  if(Geant4_VERSION VERSION_LESS 11.0)
    list(REMOVE_ITEM _along_step_filter)
  endif()
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
celeritas_add_test(grid/ElementCdfCalculator.test.cc)
celeritas_add_test(grid/GridIdFinder.test.cc)
celeritas_add_test(grid/InverseCdfFinder.test.cc)
celeritas_add_test(grid/InverseRangeCalculator.test.cc)
celeritas_add_test(grid/NonuniformGridCalculator.test.cc)
celeritas_add_test(grid/NonuniformGridBuilder.test.cc)
celeritas_add_test(grid/NonuniformGridInserter.test.cc)
celeritas_add_test(grid/RangeCalculator.test.cc)
celeritas_add_test(grid/SplineCalculator.test.cc)
celeritas_add_test(grid/UniformLogGridCalculator.test.cc)
celeritas_add_test(grid/RangeGridCalculator.test.cc)
celeritas_add_test(grid/GridInserter.test.cc)
celeritas_add_test(grid/XsCalculator.test.cc)

#-----------------------------------------------------------------------------#
# IO
celeritas_add_test(io/EventIO.test.cc ${_needs_hepmc})
celeritas_add_test(io/ImportUnits.test.cc)
celeritas_add_test(io/JsonEventIO.test.cc)
celeritas_add_test(io/RootEventIO.test.cc ${_needs_root})
celeritas_add_test(io/SeltzerBergerReader.test.cc ${_needs_geant4})

#-----------------------------------------------------------------------------#
# Inp
celeritas_add_test(inp/JsonIO.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)

#-----------------------------------------------------------------------------#
# Optical

function(celeritas_standalone_add_tests basename suite test backend)
  string(REPLACE "/" "_" _output_name "${basename}")
  set(_target "${CELERITASTEST_DIR}_${_output_name}")

  if(NOT CELERITAS_USE_Geant4 OR CELERITAS_CORE_GEO STREQUAL "Geant4")
    list(APPEND _test_props DISABLED TRUE)
  endif()

  if(NOT TARGET "${_target}")
    add_executable(${_target} "${basename}.test.cc")
    set_property(TARGET ${_target} PROPERTY OUTPUT_NAME "${_output_name}")

    celeritas_target_link_libraries(${_target}
      ${CELERITASTEST_LINK_LIBRARIES}
      ${_g4_libs}
      testcel_harness
    )
  endif()

  set(_test_name "celeritas/${basename}:${suite}.${test}:${backend}")
  add_test(
    NAME "${_test_name}"
    COMMAND "$<TARGET_FILE:${_target}>" "--gtest_filter=${suite}.${test}"
  )

  set(_env ${CELER_G4ENV})
  set(_labels "unit")

  if("${backend}" STREQUAL "cpu")
    if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
      list(APPEND _env "CELER_DISABLE_DEVICE=1")
    endif()
  elseif("${backend}" STREQUAL "gpu")
    if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
      list(APPEND _test_props RESOURCE_LOCK gpu)
      list(APPEND _labels gpu)
    else()
      list(APPEND _test_props DISABLED TRUE)
    endif()
  else()
    message(SEND_ERROR "Invalid backend type ${backend}")
  endif()

  set_tests_properties("${_test_name}" PROPERTIES
    ENVIRONMENT "${_env}"
    LABELS "${_labels}"
    ${_test_props}
  )
endfunction()

function(celeritas_add_standalone_tests basename suite test)
  foreach(_backend IN ITEMS cpu gpu)
    celeritas_standalone_add_tests(${basename} ${suite} ${test}
      ${_backend})
  endforeach()
endfunction()

celeritas_add_standalone_tests(optical/Generator LArSphereGeneratorTest primary)
celeritas_add_standalone_tests(optical/Generator LArSphereGeneratorTest direct)
celeritas_add_standalone_tests(optical/Generator LArSphereGeneratorTest offload)
celeritas_add_standalone_tests(optical/Detector DetectorTest simple)
celeritas_add_standalone_tests(optical/Detector DetectorTest stress)

celeritas_add_test(optical/Absorption.test.cc)
celeritas_add_test(optical/Cherenkov.test.cc)
celeritas_add_test(optical/ImportedModelAdapter.test.cc)
celeritas_add_test(optical/ImportedMaterials.test.cc)
celeritas_add_test(optical/ModelImporter.test.cc)
celeritas_add_test(optical/MfpBuilder.test.cc)
celeritas_add_test(optical/Mie.test.cc)
celeritas_add_test(optical/OpticalCollector.test.cc GPU ${_needs_geant4})
celeritas_add_test(optical/OpticalUtils.test.cc GPU)
celeritas_add_test(optical/Rayleigh.test.cc)
celeritas_add_test(optical/RayleighMfpCalculator.test.cc)
celeritas_add_test(optical/RoughnessCalculator.test.cc)
celeritas_add_test(optical/Scintillation.test.cc)
celeritas_add_test(optical/WavelengthShift.test.cc ${_needs_double})
celeritas_add_test(optical/Physics.test.cc)
celeritas_add_test(optical/VolumeSurfaceSelector.test.cc
    LINK_LIBRARIES testcel_geocel)
celeritas_add_test(optical/SurfacePhysics.test.cc)
celeritas_add_test(optical/ReflectionFormSampler.test.cc)
celeritas_add_test(optical/Fresnel.test.cc)
celeritas_add_test(optical/SurfacePhysicsUtils.test.cc)
celeritas_add_test(optical/TrivialSurfaceInteractor.test.cc)
celeritas_add_test(optical/SurfacePhysicsInteractionIntegration.test.cc ${_needs_geant4} ${_fails_vgsurf})
celeritas_add_test(optical/SurfacePhysicsRoughnessIntegration.test.cc
  ${_needs_geant4})
celeritas_add_test(optical/SurfacePhysicsReflectivityIntegration.test.cc ${_needs_geant4})
celeritas_add_test(optical/SurfaceImporter.test.cc ${_needs_geant4})

#-----------------------------------------------------------------------------#
# Mat
celeritas_add_device_test(mat/Material
  LINK_LIBRARIES Celeritas::ExtThrust)

#-----------------------------------------------------------------------------#
# Neutron
celeritas_add_test(neutron/NeutronElastic.test.cc ${_needs_double})
celeritas_add_test(neutron/NeutronInelastic.test.cc ${_needs_double})

#-----------------------------------------------------------------------------#
# Phys
celeritas_add_test(phys/CutoffParams.test.cc)
celeritas_add_test(phys/FourVector.test.cc)
celeritas_add_device_test(phys/Particle
  LINK_LIBRARIES Celeritas::ExtThrust)
celeritas_add_device_test(phys/Physics
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(phys/InteractionUtils.test.cc)
celeritas_add_test(phys/PhysicsStepUtils.test.cc)
celeritas_add_test(phys/PrimaryGenerator.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(phys/ProcessBuilder.test.cc ${_needs_root}
  ${_optional_geant4_env})
celeritas_add_test(phys/SurfacePhysicsMap.test.cc
  LINK_LIBRARIES testcel_geocel)

#-----------------------------------------------------------------------------#
# Random
celeritas_add_test(random/ElementSelector.test.cc)
celeritas_add_test(random/IsotopeSelector.test.cc)
celeritas_add_test(random/RngReseed.test.cc)

#-----------------------------------------------------------------------------#
# Track
celeritas_add_test(track/Sim.test.cc ${_needs_geant4})
celeritas_add_test(track/StatusChecker.test.cc GPU)
celeritas_add_test(track/TrackSort.test.cc GPU ${_needs_geant4})
celeritas_add_test(track/TrackInit.test.cc GPU)

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

add_custom_target(debug-root-test-data
  COMMENT "Debug ROOT test data"
)

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_ROOT)
  foreach(_filename "four-steel-slabs" "simple-cms")
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
    add_custom_command(
      TARGET debug-root-test-data
      POST_BUILD
      COMMAND
        "${CMAKE_COMMAND}" "-E" "env" ${CELER_G4ENV}
        "$<TARGET_FILE:celer-export-geant>"
        "${_gdml}"
        "${_basename}.geant.json"
        "${_basename}.json"
        "--gen-test"
      VERBATIM
    )
  endforeach()
endif()

#-----------------------------------------------------------------------------#
