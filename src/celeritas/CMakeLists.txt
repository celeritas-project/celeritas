#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  Celeritas::ExtDeviceApi
  nlohmann_json::nlohmann_json
  Celeritas::ExtMPI
  Celeritas::ExtOpenMP
  Celeritas::ExtThrust
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
  Celeritas::corecel
  Celeritas::geocel
)

# Core data has transient includes
if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  # NOTE: use vecgeom directly here since we know it's enabled and it's
  # necessary to make CudaRdc work downstream
  set(_core_geo_deps VecGeom::vecgeom)
elseif(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  set(_core_geo_deps Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  set(_core_geo_deps Celeritas::ExtGeant4Geo)
endif()
list(APPEND PUBLIC_DEPS ${_core_geo_deps})

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  Types.cc
  TypesIO.json.cc
  em/detail/Utils.cc
  em/params/AtomicRelaxationParams.cc
  em/params/FluctuationParams.cc
  em/params/UrbanMscParams.cc
  em/params/WentzelOKVIParams.cc
  em/params/WentzelVIMscParams.cc
  em/params/detail/MscParamsHelper.cc
  em/process/BremsstrahlungProcess.cc
  em/process/ComptonProcess.cc
  em/process/CoulombScatteringProcess.cc
  em/process/EIonizationProcess.cc
  em/process/EPlusAnnihilationProcess.cc
  em/process/GammaConversionProcess.cc
  em/process/MuIonizationProcess.cc
  em/process/MuBremsstrahlungProcess.cc
  em/process/MuPairProductionProcess.cc
  em/process/PhotoelectricProcess.cc
  em/process/RayleighProcess.cc
  ext/GeantOpticalPhysicsOptionsIO.json.cc
  ext/GeantPhysicsOptions.cc
  ext/GeantPhysicsOptionsIO.json.cc
  field/FieldDriverOptions.cc
  field/FieldDriverOptionsIO.json.cc
  field/RZMapFieldInputIO.json.cc
  field/RZMapFieldParams.cc
  field/CylMapFieldParams.cc
  field/UniformFieldParams.cc
  geo/GeoMaterialParams.cc
  global/ActionGroups.cc
  global/ActionSequence.cc
  global/CoreParams.cc
  global/CoreState.cc
  global/CoreTrackData.cc
  global/Debug.cc
  global/DebugIO.json.cc
  global/KernelContextException.cc
  global/Stepper.cc
  grid/ElementCdfCalculator.cc
  grid/NonuniformGridBuilder.cc
  grid/RangeGridCalculator.cc
  grid/TwodGridBuilder.cc
  grid/UniformGridInserter.cc
  grid/XsGridInserter.cc
  io/AtomicRelaxationReader.cc
  io/ImportData.cc
  io/ImportDataTrimmer.cc
  io/ImportMaterial.cc
  io/ImportModel.cc
  io/ImportOpticalModel.cc
  io/ImportProcess.cc
  io/ImportUnits.cc
  io/LivermorePEReader.cc
  io/NeutronXsReader.cc
  io/SeltzerBergerReader.cc
  io/detail/ImportDataConverter.cc
  mat/MaterialParams.cc
  mat/MaterialParamsOutput.cc
  mat/detail/Utils.cc
  neutron/model/CascadeOptions.cc
  neutron/model/CascadeOptionsIO.json.cc
  neutron/process/NeutronElasticProcess.cc
  neutron/process/NeutronInelasticProcess.cc
  optical/CoreParams.cc
  optical/CoreState.cc
  optical/CoreTrackData.cc
  optical/gen/CherenkovParams.cc
  optical/gen/ScintillationParams.cc
  optical/gen/GeneratorBase.cc
  optical/ImportedModelAdapter.cc
  optical/ImportedMaterials.cc
  optical/OpticalCollector.cc
  optical/MaterialParams.cc
  optical/ModelImporter.cc
  optical/PhysicsParams.cc
  optical/Transporter.cc
  optical/Types.cc
  optical/action/ActionGroups.cc
  optical/action/LocateVacanciesAction.cc
  optical/detail/OpticalLaunchAction.cc
  optical/surface/SurfacePhysicsParams.cc
  optical/surface/SurfaceSteppingAction.cc
  phys/CutoffParams.cc
  phys/detail/EnergyMaxXsCalculator.cc
  phys/GeneratorInterface.cc
  phys/GeneratorRegistry.cc
  phys/ImportedModelAdapter.cc
  phys/ImportedProcessAdapter.cc
  phys/ParticleParams.cc
  phys/ParticleParamsOutput.cc
  phys/PhysicsParams.cc
  phys/PhysicsParamsOutput.cc
  phys/PrimaryGenerator.cc
  phys/PrimaryGeneratorOptions.cc
  phys/PrimaryGeneratorOptionsIO.json.cc
  phys/Process.cc
  phys/ProcessBuilder.cc
  phys/SurfaceModel.cc
  phys/SurfacePhysicsMapBuilder.cc
  setup/Events.cc
  setup/FrameworkInput.cc
  setup/Import.cc
  setup/Model.cc
  setup/Problem.cc
  setup/StandaloneInput.cc
  setup/System.cc
  track/SimParams.cc
  track/SortTracksAction.cc
  track/TrackInitParams.cc
  user/DetectorSteps.cc
  user/ParticleTallyData.cc
  user/RootStepWriterIO.json.cc
  user/SDParams.cc
  user/SimpleCalo.cc
  user/SimpleCaloData.cc
  user/StepCollector.cc
  user/detail/StepParams.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent code
#-----------------------------------------------------------------------------#

if(CELERITAS_CORE_RNG STREQUAL "CURAND"
    OR CELERITAS_CORE_RNG STREQUAL "hipRAND")
  list(APPEND PUBLIC_DEPS Celeritas::ExtDeviceApi)
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    track/detail/Filler.cu
    alongstep/detail/AlongStepKernels.cu
  )
endif()

if(CELERITAS_USE_covfie)
  set(_celer_covfie_src
    field/CartMapFieldParams.covfie.cc
  )
  celeritas_polysource_append(_celer_covfie_src
    alongstep/AlongStepCartMapFieldMscAction)

  celeritas_add_object_library(celeritas_covfie ${_celer_covfie_src})
  celeritas_target_link_libraries(celeritas_covfie
    PRIVATE
      Celeritas::BuildFlags
      Celeritas::geocel
      ${_core_geo_deps}
      Celeritas::Extcovfie
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_covfie>)
  list(APPEND PRIVATE_DEPS celeritas_covfie)
else()
  celeritas_polysource_append(SOURCES
  alongstep/AlongStepCartMapFieldMscAction)
endif()

if(CELERITAS_USE_Geant4)
  set(_cg4_sources
    ext/EmPhysicsList.cc
    ext/FtfpBertPhysicsList.cc
    ext/GeantImporter.cc
    ext/GeantSd.cc
    ext/GeantSdOutput.cc
    ext/GeantSetup.cc
    ext/GeantVolumeMapper.cc
    ext/detail/EmStandardPhysics.cc
    ext/detail/GeantBremsstrahlungProcess.cc
    ext/detail/GeantMicroXsCalculator.cc
    ext/detail/GeantModelImporter.cc
    ext/detail/GeantOpticalModelImporter.cc
    ext/detail/GeantProcessImporter.cc
    ext/detail/GeantSurfacePhysicsHelper.cc
    ext/detail/GeantSurfacePhysicsLoader.cc
    ext/detail/GeoOpticalIdMap.cc
    ext/detail/HitProcessor.cc
    ext/detail/LevelTouchableUpdater.cc
    ext/detail/MuHadEmStandardPhysics.cc
    ext/detail/NaviTouchableUpdater.cc
    ext/detail/OpticalPhysics.cc
    ext/detail/SensDetInserter.cc
    ext/detail/TouchableUpdaterInterface.cc
    ext/detail/TrackProcessor.cc
  )

  celeritas_get_g4libs(_cg4_libs
    global geometry materials processes run physicslists tasking
  )
  celeritas_add_object_library(celeritas_geant4 ${_cg4_sources})
  celeritas_target_link_libraries(celeritas_geant4
    PRIVATE
      Celeritas::BuildFlags
      Celeritas::geocel
      ${_core_geo_deps} # needed by GeantSd and related
      XercesC::XercesC
      nlohmann_json::nlohmann_json
      ${_cg4_libs}
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_geant4>)
  list(APPEND PRIVATE_DEPS celeritas_geant4)
endif()

if(CELERITAS_USE_HepMC3)
  celeritas_add_object_library(celeritas_hepmc
    io/EventReader.cc
    io/EventWriter.cc
  )
  target_link_libraries(celeritas_hepmc
    PRIVATE Celeritas::corecel HepMC3::HepMC3
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_hepmc>)
  list(APPEND PRIVATE_DEPS celeritas_hepmc)
endif()

if(CELERITAS_USE_ROOT)
  # Use directory includes because ROOT has trouble with build/install
  # interface dependencies propagated through corecel.
  # This is safe here as it is only adding project-local paths that are
  # identical to those set in corecel's usage requirements.
  include_directories(
    "${PROJECT_SOURCE_DIR}/src"
    "${CELERITAS_HEADER_CONFIG_DIRECTORY}"
  )

  # Set the CMAKE output directory locally to inform ROOT where we put our
  # libs. Safe against overriding project settings as the celeritas_add_...
  # functions set this to the same value for our targets.
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CELERITAS_LIBRARY_OUTPUT_DIRECTORY})

  # Generate the dictionary source file
  root_generate_dictionary(CeleritasRootInterface
    "celeritas/io/ImportData.hh"
    "celeritas/io/ImportElement.hh"
    "celeritas/io/ImportMaterial.hh"
    "celeritas/io/ImportParticle.hh"
    "celeritas/io/ImportPhysicsTable.hh"
    "celeritas/io/ImportProcess.hh"
    "celeritas/io/ImportUnits.hh"
    "celeritas/io/ImportVolume.hh"
    "celeritas/io/EventData.hh"
    "celeritas/inp/Grid.hh"
    NOINSTALL
    MODULE celeritas
    LINKDEF "${CMAKE_CURRENT_SOURCE_DIR}/ext/RootInterfaceLinkDef.h"
  )
  celeritas_add_object_library(celeritas_root
    ext/RootExporter.cc
    ext/RootImporter.cc
    ext/RootJsonDumper.cc
    ext/ScopedRootErrorHandler.cc
    ext/RootUniquePtr.root.cc
    ext/RootFileManager.cc
    io/RootCoreParamsOutput.cc
    io/RootEventReader.cc
    io/RootEventWriter.cc
    phys/RootEventSampler.cc
    user/RootStepWriter.cc
    "${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx"
  )

  target_link_libraries(celeritas_root
    PRIVATE Celeritas::BuildFlags ROOT::Core ROOT::Tree ROOT::RIO
  )

  # Install the rootmap/pcm files needed for users or downstream apps to use
  # Celeritas ROOT interfaces
  set(_lib_prefix
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}celeritas"
  )
  install(FILES
    "${_lib_prefix}.rootmap"
    "${_lib_prefix}_rdict.pcm"
    COMPONENT runtime
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )

  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_root>)
  list(APPEND PRIVATE_DEPS celeritas_root)
endif()

#-----------------------------------------------------------------------------#
# Optional CUDA code
#-----------------------------------------------------------------------------#

macro(celeritas_polysource filename_we)
  celeritas_polysource_append(SOURCES ${filename_we})
endmacro()

celeritas_polysource(alongstep/AlongStepGeneralLinearAction)
celeritas_polysource(alongstep/AlongStepNeutralAction)
celeritas_polysource(alongstep/AlongStepUniformMscAction)
celeritas_polysource(alongstep/AlongStepRZMapFieldMscAction)
celeritas_polysource(alongstep/AlongStepCylMapFieldMscAction)
celeritas_polysource(em/model/BetheHeitlerModel)
celeritas_polysource(em/model/BetheBlochModel)
celeritas_polysource(em/model/BraggModel)
celeritas_polysource(em/model/EPlusGGModel)
celeritas_polysource(em/model/ICRU73QOModel)
celeritas_polysource(em/model/KleinNishinaModel)
celeritas_polysource(em/model/LivermorePEModel)
celeritas_polysource(em/model/MollerBhabhaModel)
celeritas_polysource(em/model/MuBetheBlochModel)
celeritas_polysource(em/model/MuBremsstrahlungModel)
celeritas_polysource(em/model/MuPairProductionModel)
celeritas_polysource(em/model/RayleighModel)
celeritas_polysource(em/model/RelativisticBremModel)
celeritas_polysource(em/model/SeltzerBergerModel)
celeritas_polysource(em/model/CoulombScatteringModel)
celeritas_polysource(geo/detail/BoundaryAction)
celeritas_polysource(global/detail/KillActive)
celeritas_polysource(global/detail/TrackSlotUtils)
celeritas_polysource(neutron/model/ChipsNeutronElasticModel)
celeritas_polysource(neutron/model/NeutronInelasticModel)
celeritas_polysource(optical/model/AbsorptionModel)
celeritas_polysource(optical/model/RayleighModel)
celeritas_polysource(optical/model/WavelengthShiftModel)
celeritas_polysource(optical/action/AlongStepAction)
celeritas_polysource(optical/action/DiscreteSelectAction)
celeritas_polysource(optical/action/PreStepAction)
celeritas_polysource(optical/action/TrackingCutAction)
celeritas_polysource(optical/action/detail/TrackInitAlgorithms)
celeritas_polysource(optical/gen/GeneratorAction)
celeritas_polysource(optical/gen/OffloadAction)
celeritas_polysource(optical/gen/OffloadGatherAction)
celeritas_polysource(optical/gen/PrimaryGeneratorAction)
celeritas_polysource(optical/gen/detail/GeneratorAlgorithms)
celeritas_polysource(optical/gen/detail/OffloadAlgorithms)
celeritas_polysource(optical/surface/BoundaryAction)
celeritas_polysource(phys/detail/DiscreteSelectAction)
celeritas_polysource(phys/detail/PreStepAction)
celeritas_polysource(phys/detail/TrackingCutAction)
celeritas_polysource(random/RngReseed)
celeritas_polysource(track/detail/TrackInitAlgorithms)
celeritas_polysource(track/detail/TrackSortUtils)
celeritas_polysource(track/ExtendFromPrimariesAction)
celeritas_polysource(track/ExtendFromSecondariesAction)
celeritas_polysource(track/InitializeTracksAction)
celeritas_polysource(track/StatusChecker)
celeritas_polysource(user/ActionDiagnostic)
celeritas_polysource(user/DetectorSteps)
celeritas_polysource(user/SlotDiagnostic)
celeritas_polysource(user/StepDiagnostic)
celeritas_polysource(user/detail/SimpleCaloImpl)
celeritas_polysource(user/detail/StepGatherAction)

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(celeritas ${SOURCES})

celeritas_target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
