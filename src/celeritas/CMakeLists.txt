#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::DeviceToolkit)
set(PUBLIC_DEPS Celeritas::corecel)


# Add an object library to limit the propagation of includes to the rest of the library.
macro(celeritas_add_object_library target)
  add_library(${target} OBJECT ${ARGN})
  install(TARGETS ${target}
    EXPORT celeritas-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT runtime
  )
endmacro()

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  em/AtomicRelaxationParams.cc
  em/FluctuationParams.cc
  em/detail/Utils.cc
  em/model/BetheHeitlerModel.cc
  em/model/CombinedBremModel.cc
  em/model/EPlusGGModel.cc
  em/model/KleinNishinaModel.cc
  em/model/LivermorePEModel.cc
  em/model/MollerBhabhaModel.cc
  em/model/MuBremsstrahlungModel.cc
  em/model/RayleighModel.cc
  em/model/RelativisticBremModel.cc
  em/model/SeltzerBergerModel.cc
  em/model/UrbanMscModel.cc
  em/process/BremsstrahlungProcess.cc
  em/process/ComptonProcess.cc
  em/process/EIonizationProcess.cc
  em/process/EPlusAnnihilationProcess.cc
  em/process/GammaConversionProcess.cc
  em/process/MultipleScatteringProcess.cc
  em/process/PhotoelectricProcess.cc
  em/process/RayleighProcess.cc
  # ext/MpiCommunicator.cc
  # ext/ScopedMpiInit.cc
  geo/GeoMaterialParams.cc
  global/ActionInterface.cc
  global/ActionManager.cc
  global/ActionManagerOutput.cc
  global/CoreParams.cc
  global/Stepper.cc
  grid/ValueGridBuilder.cc
  grid/ValueGridInserter.cc
  grid/ValueGridInterface.cc
  io/AtomicRelaxationReader.cc
  io/ImportData.cc
  io/ImportPhysicsTable.cc
  io/ImportPhysicsVector.cc
  io/ImportProcess.cc
  io/LivermorePEReader.cc
  io/SeltzerBergerReader.cc
  mat/MaterialParams.cc
  mat/detail/Utils.cc
  phys/CutoffParams.cc
  phys/ImportedModelAdapter.cc
  phys/ImportedProcessAdapter.cc
  phys/ParticleParams.cc
  phys/PhysicsParams.cc
  phys/PhysicsParamsOutput.cc
  phys/Process.cc
  phys/ProcessBuilder.cc
  random/CuHipRngData.cc
  random/XorwowRngData.cc
  random/XorwowRngParams.cc
  track/TrackInitParams.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent code
#-----------------------------------------------------------------------------#


if(CELERITAS_RNG STREQUAL "CURAND")
  list(APPEND PUBLIC_DEPS Celeritas::DeviceToolkit)
endif()

if(CELERITAS_USE_Geant4)
  celeritas_add_object_library(celeritas_geant4
    ext/LoadGdml.cc
    ext/GeantImporter.cc
    ext/GeantSetup.cc
    ext/detail/GeantBremsstrahlungProcess.cc
    ext/detail/GeantExceptionHandler.cc
    ext/detail/GeantLoggerAdapter.cc
    ext/detail/GeantPhysicsList.cc
    ext/detail/ImportProcessConverter.cc
  )
  target_link_libraries(celeritas_geant4
    PRIVATE Celeritas::corecel XercesC::XercesC ${Geant4_LIBRARIES}
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_geant4>)
  list(APPEND PRIVATE_DEPS celeritas_geant4)
endif()

if(CELERITAS_USE_HepMC3)
  celeritas_add_object_library(celeritas_hepmc
    io/EventReader.cc
  )
  target_link_libraries(celeritas_hepmc
    PRIVATE Celeritas::corecel HepMC3::HepMC3
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_hepmc>)
  list(APPEND PRIVATE_DEPS celeritas_hepmc)
else()
  list(APPEND SOURCES
    io/EventReader.nohepmc.cc
  )
endif()

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    ext/GeantPhysicsOptionsIO.json.cc
    field/FieldDriverOptionsIO.json.cc
    phys/PrimaryGeneratorOptionsIO.json.cc
  )
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

if(CELERITAS_USE_ROOT)
  # Call the ROOT library generation inside a function to prevent ROOT-defined
  # directory properties from escaping
  function(celeritas_root_library objlib root_tgt)
    include(${ROOT_USE_FILE})
    # Use directory includes because ROOT has trouble with build/install
    # interface dependencies propagated through corecel
    include_directories(
      "${PROJECT_SOURCE_DIR}/src"
      "${CELERITAS_HEADER_CONFIG_DIRECTORY}"
    )

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
    celeritas_add_object_library(${objlib}
      ext/RootExporter.cc
      ext/RootImporter.cc
      ext/detail/TFileUniquePtr.root.cc
      "${CMAKE_CURRENT_BINARY_DIR}/${root_tgt}.cxx"
    )
    target_link_libraries(${objlib}
      PRIVATE Celeritas::corecel ROOT::Core ROOT::Tree
    )
  endfunction()

  celeritas_root_library(celeritas_root CeleritasRootInterface)
  list(APPEND SOURCES $<TARGET_OBJECTS:celeritas_root>)
  list(APPEND PRIVATE_DEPS celeritas_root)
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    ext/VecgeomParams.cc
    ext/detail/VecgeomNavCollection.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml)
  # This needs to be public because its might be needed
  # to resolve the symbols generate by the `nvcc -dlink` of
  # one of the executable.
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
else()
  list(APPEND PUBLIC_DEPS Celeritas::orange)
endif()

#-----------------------------------------------------------------------------#
# Auto-generated code
#-----------------------------------------------------------------------------#

# Generate model launcher and track initialization code for host/device
macro(celeritas_gen_interactor class func)
  celeritas_gen(SOURCES
    "gen-interactor.py" "em/generated/${class}Interact"
    --class ${class}  --func ${func}
  )
endmacro()

macro(celeritas_gen_action basedir class func)
  celeritas_gen(SOURCES
    "gen-action.py" "${basedir}/generated/${class}"
    --class ${class} --func ${func}
  )
endmacro()

macro(celeritas_gen_trackinit class)
  celeritas_gen(SOURCES
    "gen-trackinit.py" "track/generated/${class}"
    --class ${class}
  )
endmacro()

macro(celeritas_polysource filename_we)
  list(APPEND SOURCES "${filename_we}.cc")
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND SOURCES "${filename_we}.cu")
  endif()
endmacro()

celeritas_gen_interactor("BetheHeitler" "bethe_heitler")
celeritas_gen_interactor("CombinedBrem" "combined_brem")
celeritas_gen_interactor("EPlusGG" "eplusgg")
celeritas_gen_interactor("KleinNishina" "klein_nishina")
celeritas_gen_interactor("LivermorePE" "livermore_pe")
celeritas_gen_interactor("MollerBhabha" "moller_bhabha")
celeritas_gen_interactor("MuBremsstrahlung" "mu_bremsstrahlung")
celeritas_gen_interactor("Rayleigh" "rayleigh")
celeritas_gen_interactor("RelativisticBrem" "relativistic_brem")
celeritas_gen_interactor("SeltzerBerger" "seltzer_berger")

celeritas_gen_action("phys" "DiscreteSelectAction" "discrete_select")
celeritas_gen_action("phys" "PreStepAction" "pre_step")
celeritas_gen_action("geo" "BoundaryAction" "boundary")

celeritas_gen_trackinit("InitTracks")
celeritas_gen_trackinit("LocateAlive")
celeritas_gen_trackinit("ProcessPrimaries")
celeritas_gen_trackinit("ProcessSecondaries")

celeritas_polysource(global/alongstep/AlongStepGeneralLinearAction)
celeritas_polysource(global/alongstep/AlongStepNeutralAction)
celeritas_polysource(global/alongstep/AlongStepUniformMscAction)
celeritas_polysource(random/detail/CuHipRngStateInit)
celeritas_polysource(track/detail/TrackInitAlgorithms)

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

celeritas_target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
# Install
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_ROOT)
  # Install the rootmap/pcm files needed for users or downstream apps to use
  # Celeritas ROOT interfaces
  set(_lib_prefix
    "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}celeritas"
  )
  install(FILES
    "${_lib_prefix}.rootmap"
    "${_lib_prefix}_rdict.pcm"
    COMPONENT runtime
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

#-----------------------------------------------------------------------------#
