#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::DeviceToolkit)
set(PUBLIC_DEPS Celeritas::corecel)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  Types.cc
  em/AtomicRelaxationParams.cc
  em/FluctuationParams.cc
  em/UrbanMscParams.cc
  em/detail/Utils.cc
  em/process/BremsstrahlungProcess.cc
  em/process/ComptonProcess.cc
  em/process/CoulombScatteringProcess.cc
  em/process/EIonizationProcess.cc
  em/process/EPlusAnnihilationProcess.cc
  em/process/GammaConversionProcess.cc
  em/process/PhotoelectricProcess.cc
  em/process/RayleighProcess.cc
  ext/GeantPhysicsOptions.cc
  field/RZMapFieldParams.cc
  geo/GeoMaterialParams.cc
  geo/GeoParamsOutput.cc
  global/ActionInterface.cc
  global/ActionRegistry.cc
  global/ActionRegistryOutput.cc
  global/CoreParams.cc
  global/CoreState.cc
  global/CoreTrackData.cc
  global/KernelContextException.cc
  global/Stepper.cc
  global/detail/ActionSequence.cc
  global/detail/PinnedAllocator.cc
  grid/ValueGridBuilder.cc
  grid/ValueGridData.cc
  grid/ValueGridInserter.cc
  grid/VectorUtils.cc
  io/AtomicRelaxationReader.cc
  io/ImportModel.cc
  io/ImportPhysicsTable.cc
  io/ImportPhysicsVector.cc
  io/ImportProcess.cc
  io/LivermorePEReader.cc
  io/SeltzerBergerReader.cc
  mat/MaterialParams.cc
  mat/MaterialParamsOutput.cc
  mat/detail/Utils.cc
  phys/CutoffParams.cc
  phys/ImportedModelAdapter.cc
  phys/ImportedProcessAdapter.cc
  phys/ParticleParams.cc
  phys/ParticleParamsOutput.cc
  phys/PhysicsParams.cc
  phys/PhysicsParamsOutput.cc
  phys/PrimaryGenerator.cc
  phys/PrimaryGeneratorOptions.cc
  phys/Process.cc
  phys/ProcessBuilder.cc
  random/CuHipRngData.cc
  random/XorwowRngData.cc
  random/XorwowRngParams.cc
  track/SimParams.cc
  track/SortTracksAction.cc
  track/TrackInitParams.cc
  user/ParticleTallyData.cc
  user/DetectorSteps.cc
  user/SimpleCalo.cc
  user/SimpleCaloData.cc
  user/StepCollector.cc
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
    global/alongstep/detail/AlongStepKernels.cu
  )
endif()

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_Geant4)
  set( _cg4vg_sources
    ext/g4vg/Converter.cc
    ext/g4vg/LogicalVolumeConverter.cc
    ext/g4vg/SolidConverter.cc
  )
  set(_cg4vg_libs Celeritas::corecel VecGeom::vecgeom ${Geant4_LIBRARIES})

  celeritas_add_object_library(celeritas_g4vg ${_cg4vg_sources})
  celeritas_target_link_libraries(celeritas_g4vg PRIVATE ${_cg4vg_libs})

  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_g4vg>)
  list(APPEND PRIVATE_DEPS celeritas_g4vg)
endif()

if(CELERITAS_USE_Geant4)
  set(_cg4_sources
    ext/GeantGeoParams.cc
    ext/GeantGeoUtils.cc
    ext/GeantImporter.cc
    ext/GeantSetup.cc
    ext/GeantVolumeMapper.cc
    ext/ScopedGeantExceptionHandler.cc
    ext/ScopedGeantLogger.cc
    ext/detail/GeantBremsstrahlungProcess.cc
    ext/detail/GeantGeoNavCollection.cc
    ext/detail/GeantMicroXsCalculator.cc
    ext/detail/GeantModelImporter.cc
    ext/detail/GeantPhysicsList.cc
    ext/detail/GeantProcessImporter.cc
    ext/detail/GeantVolumeVisitor.cc
  )
  set(_cg4_libs Celeritas::corecel XercesC::XercesC ${Geant4_LIBRARIES})

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

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    ext/GeantPhysicsOptionsIO.json.cc
    field/FieldDriverOptionsIO.json.cc
    field/RZMapFieldInputIO.json.cc
    phys/PrimaryGeneratorOptionsIO.json.cc
    user/RootStepWriterIO.json.cc
  )
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportData.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportElement.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportMaterial.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportParticle.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportPhysicsTable.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportPhysicsVector.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportProcess.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/ImportVolume.hh"
    NOINSTALL
    MODULE celeritas
    LINKDEF "${CMAKE_CURRENT_SOURCE_DIR}/ext/RootInterfaceLinkDef.h"
  )
  celeritas_add_object_library(celeritas_root
    ext/RootExporter.cc
    ext/RootImporter.cc
    ext/ScopedRootErrorHandler.cc
    ext/RootUniquePtr.root.cc
    ext/RootFileManager.cc
    user/RootStepWriter.cc
    io/RootEventReader.cc
    io/RootEventWriter.cc
    "${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx"
  )

  target_link_libraries(celeritas_root
    PRIVATE Celeritas::corecel ROOT::Core ROOT::Tree
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

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    ext/VecgeomParams.cc
    ext/detail/VecgeomNavCollection.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml VecGeom::vecgeom)
endif()

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
  # This needs to be public because its might be needed
  # to resolve the symbols generate by the `nvcc -dlink` of
  # one of the executable.
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
endif()

if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  list(APPEND PUBLIC_DEPS Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  # Core data has transient includes into vecgeom
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
 list(APPEND PUBLIC_DEPS ${Geant4_LIBRARIES})
endif()

macro(celeritas_polysource filename_we)
  list(APPEND SOURCES "${filename_we}.cc")
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND SOURCES "${filename_we}.cu")
  endif()
endmacro()

#-----------------------------------------------------------------------------#
# Optional CUDA code
#-----------------------------------------------------------------------------#

celeritas_polysource(em/model/BetheHeitlerModel)
celeritas_polysource(em/model/CombinedBremModel)
celeritas_polysource(em/model/EPlusGGModel)
celeritas_polysource(em/model/KleinNishinaModel)
celeritas_polysource(em/model/LivermorePEModel)
celeritas_polysource(em/model/MollerBhabhaModel)
celeritas_polysource(em/model/MuBremsstrahlungModel)
celeritas_polysource(em/model/RayleighModel)
celeritas_polysource(em/model/RelativisticBremModel)
celeritas_polysource(em/model/SeltzerBergerModel)
celeritas_polysource(em/model/WentzelModel)
celeritas_polysource(geo/detail/BoundaryAction)
celeritas_polysource(global/alongstep/AlongStepGeneralLinearAction)
celeritas_polysource(global/alongstep/AlongStepNeutralAction)
celeritas_polysource(global/alongstep/AlongStepUniformMscAction)
celeritas_polysource(global/alongstep/AlongStepRZMapFieldMscAction)
celeritas_polysource(phys/detail/DiscreteSelectAction)
celeritas_polysource(phys/detail/PreStepAction)
celeritas_polysource(random/detail/CuHipRngStateInit)
celeritas_polysource(track/detail/TrackInitAlgorithms)
celeritas_polysource(track/detail/TrackSortUtils)
celeritas_polysource(track/ExtendFromPrimariesAction)
celeritas_polysource(track/ExtendFromSecondariesAction)
celeritas_polysource(track/InitializeTracksAction)
celeritas_polysource(user/ActionDiagnostic)
celeritas_polysource(user/DetectorSteps)
celeritas_polysource(user/StepDiagnostic)
celeritas_polysource(user/detail/SimpleCaloImpl)
celeritas_polysource(user/detail/StepGatherAction)

#-----------------------------------------------------------------------------#
# Auto-generated code
#-----------------------------------------------------------------------------#

# Generate model executor and track initialization code for host/device
macro(celeritas_gen_interactor class func)
  celeritas_gen(SOURCES
    "gen-interactor.py" "em/generated/${class}Interact"
    --class ${class}  --func ${func}
  )
endmacro()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(celeritas ${SOURCES})

if(NOT CELERITAS_USE_OpenMP
    AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
         OR CMAKE_CXX_COMPILER_ID MATCHES "Clang$"))
  celeritas_target_compile_options(celeritas
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
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
 
 
