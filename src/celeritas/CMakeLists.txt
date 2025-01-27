#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::DeviceToolkit nlohmann_json::nlohmann_json)
set(PUBLIC_DEPS Celeritas::corecel Celeritas::geocel)

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
  ext/GeantOpticalPhysicsOptions.cc
  ext/GeantOpticalPhysicsOptionsIO.json.cc
  ext/GeantPhysicsOptions.cc
  ext/GeantPhysicsOptionsIO.json.cc
  field/FieldDriverOptions.cc
  field/FieldDriverOptionsIO.json.cc
  field/RZMapFieldInputIO.json.cc
  field/RZMapFieldParams.cc
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
  grid/GenericGridBuilder.cc
  grid/TwodGridBuilder.cc
  grid/ValueGridBuilder.cc
  grid/ValueGridInserter.cc
  grid/ValueGridType.cc
  io/AtomicRelaxationReader.cc
  io/ImportData.cc
  io/ImportDataTrimmer.cc
  io/ImportMaterial.cc
  io/ImportModel.cc
  io/ImportOpticalModel.cc
  io/ImportPhysicsTable.cc
  io/ImportPhysicsVector.cc
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
  optical/CherenkovParams.cc
  optical/CoreParams.cc
  optical/CoreState.cc
  optical/CoreTrackData.cc
  optical/ImportedModelAdapter.cc
  optical/ImportedMaterials.cc
  optical/OpticalCollector.cc
  optical/MaterialParams.cc
  optical/ModelImporter.cc
  optical/TrackInitParams.cc
  optical/ScintillationParams.cc
  optical/WavelengthShiftParams.cc
  optical/action/ActionGroups.cc
  optical/action/LocateVacanciesAction.cc
  optical/detail/OffloadParams.cc
  optical/detail/OpticalLaunchAction.cc
  phys/CutoffParams.cc
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
  random/CuHipRngData.cc
  random/CuHipRngParams.cc
  random/XorwowRngData.cc
  random/XorwowRngParams.cc
  setup/Import.cc
  setup/System.cc
  track/SimParams.cc
  track/SortTracksAction.cc
  track/TrackInitParams.cc
  user/DetectorSteps.cc
  user/ParticleTallyData.cc
  user/RootStepWriterIO.json.cc
  user/SimpleCalo.cc
  user/SimpleCaloData.cc
  user/StepCollector.cc
  user/detail/StepParams.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent code
#-----------------------------------------------------------------------------#

if(CELERITAS_CORE_RNG STREQUAL "CURAND")
  list(APPEND PUBLIC_DEPS Celeritas::DeviceToolkit)
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    track/detail/Filler.cu
    alongstep/detail/AlongStepKernels.cu
  )
endif()

if(CELERITAS_USE_Geant4)
  set(_cg4_sources
    ext/GeantImporter.cc
    ext/GeantSetup.cc
    ext/GeantVolumeMapper.cc
    ext/detail/CelerEmPhysicsList.cc
    ext/detail/CelerEmStandardPhysics.cc
    ext/detail/CelerFTFPBert.cc
    ext/detail/CelerOpticalPhysics.cc
    ext/detail/GeantBremsstrahlungProcess.cc
    ext/detail/GeantMicroXsCalculator.cc
    ext/detail/GeantModelImporter.cc
    ext/detail/GeantOpticalModelImporter.cc
    ext/detail/GeantProcessImporter.cc
    ext/detail/GeoOpticalIdMap.cc
    ext/detail/MuHadEmStandardPhysics.cc
  )
  celeritas_get_g4libs(_cg4_libs
    global geometry materials processes run physicslists tasking
  )
  list(APPEND _cg4_libs Celeritas::corecel XercesC::XercesC)

  celeritas_add_object_library(celeritas_geant4 ${_cg4_sources})
  celeritas_target_link_libraries(celeritas_geant4 PRIVATE ${_cg4_libs})

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

if(CELERITAS_USE_MPI)
  list(APPEND PRIVATE_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

if(CELERITAS_USE_ROOT)
  # Use directory includes because ROOT has trouble with build/install
  # interface dependencies propagated through corecel.
  # This is safe here as it is only adding project-local paths that are
  # identical to those set in corecel's usage requirments.
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
    "celeritas/io/ImportPhysicsVector.hh"
    "celeritas/io/ImportProcess.hh"
    "celeritas/io/ImportUnits.hh"
    "celeritas/io/ImportVolume.hh"
    "celeritas/io/EventData.hh"
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
    io/RootEventReader.cc
    io/RootEventWriter.cc
    phys/RootEventSampler.cc
    user/RootStepWriter.cc
    "${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx"
  )

  target_link_libraries(celeritas_root
    PRIVATE Celeritas::corecel ROOT::Core ROOT::Tree ROOT::RIO
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

# Core data has transient includes
if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
elseif(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  list(APPEND PUBLIC_DEPS Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
 list(APPEND PUBLIC_DEPS ${Geant4_LIBRARIES})
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
celeritas_polysource(em/model/BetheHeitlerModel)
celeritas_polysource(em/model/BetheBlochModel)
celeritas_polysource(em/model/BraggModel)
celeritas_polysource(em/model/CombinedBremModel)
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
celeritas_polysource(optical/action/AlongStepAction)
celeritas_polysource(optical/action/BoundaryAction)
celeritas_polysource(optical/action/InitializeTracksAction)
celeritas_polysource(optical/action/PreStepAction)
celeritas_polysource(optical/action/TrackingCutAction)
celeritas_polysource(optical/action/detail/TrackInitAlgorithms)
celeritas_polysource(optical/detail/CherenkovGeneratorAction)
celeritas_polysource(optical/detail/CherenkovOffloadAction)
celeritas_polysource(optical/detail/OpticalGenAlgorithms)
celeritas_polysource(optical/detail/OffloadGatherAction)
celeritas_polysource(optical/detail/ScintGeneratorAction)
celeritas_polysource(optical/detail/ScintOffloadAction)
celeritas_polysource(phys/detail/DiscreteSelectAction)
celeritas_polysource(phys/detail/PreStepAction)
celeritas_polysource(phys/detail/TrackingCutAction)
celeritas_polysource(random/RngReseed)
celeritas_polysource(random/detail/CuHipRngStateInit)
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

if(NOT CELERITAS_USE_OpenMP
    AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
         OR CMAKE_CXX_COMPILER_ID MATCHES "Clang$"))
  celeritas_target_compile_options(celeritas
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>"
  )
endif()
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Inheriting via dominance is correct behavior
  celeritas_target_compile_options(celeritas
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/wd4267$<SEMICOLON>/wd4250>"
  )
endif()

celeritas_target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
