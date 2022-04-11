#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# CONFIGURE FILE
#----------------------------------------------------------------------------#
set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

# Define a numeric table of options for the default runtime RNG.
# Start counter from 1 because undefined macros have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
set(CELERITAS_RNG_MACROS)
set(_rng_counter 1)
foreach(_rng IN LISTS CELERITAS_RNG_OPTIONS)
  list(APPEND CELERITAS_RNG_MACROS
    "#define CELERITAS_RNG_${_rng} ${_rng_counter}"
  )
  math(EXPR _rng_counter "${_rng_counter} + 1")
endforeach()

string(JOIN "\n" CELERITAS_RNG_MACROS
  ${CELERITAS_RNG_MACROS}
  "#define CELERITAS_RNG CELERITAS_RNG_${CELERITAS_RNG}"
)
configure_file("celeritas_config.h.in" "celeritas_config.h" @ONLY)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/celeritas_config.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS)

# Version information
configure_file("celeritas_version.cc.in" "celeritas_version.cc" @ONLY)
list(APPEND SOURCES
  "${CMAKE_CURRENT_BINARY_DIR}/celeritas_version.cc"
)

# Generate model launcher and track initialization code for host/device
set(_gen_sources)
macro(celeritas_gen_interactor class func)
  celeritas_gen(_gen_sources
    "gen-interactor.py" "physics/em/generated/${class}Interact"
    --class ${class}  --func ${func}
  )
endmacro()
macro(celeritas_gen_trackinit class)
  celeritas_gen(_gen_sources
    "gen-trackinit.py" "sim/generated/${class}"
    --class ${class}
  )
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

celeritas_gen_trackinit("InitTracks")
celeritas_gen_trackinit("LocateAlive")
celeritas_gen_trackinit("ProcessPrimaries")
celeritas_gen_trackinit("ProcessSecondaries")

list(APPEND SOURCES ${_gen_sources})

# Main library
list(APPEND SOURCES
  base/Assert.cc
  base/ColorUtils.cc
  base/Copier.cc
  base/DeviceAllocation.cc
  base/ScopedTimeAndRedirect.cc
  base/StringUtils.cc
  comm/KernelDiagnostics.cc
  base/ScopedStreamRedirect.cc
  base/TypeDemangler.cc
  base/VectorUtils.cc
  base/detail/ReprImpl.cc
  comm/Communicator.cc
  comm/Device.cc
  comm/Environment.cc
  comm/Logger.cc
  comm/LoggerTypes.cc
  comm/ScopedMpiInit.cc
  comm/ScopedSignalHandler.cc
  comm/detail/LoggerMessage.cc
  geometry/GeoMaterialParams.cc
  orange/OrangeParams.cc
  orange/Types.cc
  orange/construct/SurfaceInserter.cc
  orange/construct/VolumeInserter.cc
  orange/surfaces/SurfaceIO.cc
  io/ImportProcess.cc
  io/ImportPhysicsTable.cc
  io/ImportPhysicsVector.cc
  io/AtomicRelaxationReader.cc
  io/LivermorePEReader.cc
  io/SeltzerBergerReader.cc
  physics/base/CutoffParams.cc
  physics/base/ImportedProcessAdapter.cc
  physics/base/Model.cc
  physics/base/ParticleParams.cc
  physics/base/PhysicsParams.cc
  physics/base/Process.cc
  physics/em/AtomicRelaxationParams.cc
  physics/em/BetheHeitlerModel.cc
  physics/em/BremsstrahlungProcess.cc
  physics/em/CombinedBremModel.cc
  physics/em/ComptonProcess.cc
  physics/em/PhotoelectricProcess.cc
  physics/em/LivermorePEModel.cc
  physics/em/EIonizationProcess.cc
  physics/em/EPlusAnnihilationProcess.cc
  physics/em/EPlusGGModel.cc
  physics/em/GammaConversionProcess.cc
  physics/em/KleinNishinaModel.cc
  physics/em/MollerBhabhaModel.cc
  physics/em/MuBremsstrahlungModel.cc
  physics/em/MultipleScatteringProcess.cc
  physics/em/RayleighModel.cc
  physics/em/RayleighProcess.cc
  physics/em/RelativisticBremModel.cc
  physics/em/SeltzerBergerModel.cc
  physics/em/UrbanMscModel.cc
  physics/em/detail/Utils.cc
  physics/grid/ValueGridBuilder.cc
  physics/grid/ValueGridInserter.cc
  physics/grid/ValueGridInterface.cc
  physics/material/MaterialParams.cc
  physics/material/detail/Utils.cc
  random/CuHipRngData.cc
  random/XorwowRngData.cc
  random/XorwowRngParams.cc
  random/detail/CuHipRngStateInit.cc
  sim/TrackInitParams.cc
  sim/detail/TrackInitAlgorithms.cc
)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    base/KernelParamCalculator.device.cc
  )
  set(DEVICE_SOURCES
    base/detail/Filler.cu
    random/detail/CuHipRngStateInit.cu
    sim/detail/TrackInitAlgorithms.cu
  )
endif()

if(CELERITAS_USE_CUDA)
  if(CELERITAS_RNG STREQUAL "CURAND")
    list(APPEND PUBLIC_DEPS CUDA::cudart)
  else()
    list(APPEND PRIVATE_DEPS CUDA::cudart)
  endif()
elseif(CELERITAS_USE_HIP)
  set_source_files_properties(
    ${DEVICE_SOURCES}
    PROPERTIES LANGUAGE HIP
  )
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND SOURCES
    io/EventReader.cc
  )
  list(APPEND PRIVATE_DEPS HepMC3::HepMC3)
else()
  list(APPEND SOURCES
    io/EventReader.nohepmc.cc
  )
endif()

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    comm/DeviceIO.json.cc
    comm/KernelDiagnosticsIO.json.cc
    orange/construct/SurfaceInputIO.json.cc
    orange/construct/VolumeInputIO.json.cc
  )
  list(APPEND PUBLIC_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_MPI)
  list(APPEND PUBLIC_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    vecgeom/VecgeomParams.cc
    vecgeom/detail/VecgeomNavCollection.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml)
  # This needs to be public because its might be needed
  # to resolve the symbol generate by the `nvcc -dlink` of
  # one of the executable.
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
endif()

celeritas_add_library(celeritas ${SOURCES} ${DEVICE_SOURCES})
celeritas_strip_alias(celeritas_target celeritas)
add_library(Celeritas::Core ALIAS ${celeritas_target})

if(CELERITAS_USE_ROOT AND NOT BUILD_SHARED_LIBS)
  # Must build with -fPIC to link with ROOT
  set_target_properties(celeritas
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )
endif()

if(OpenMP_FOUND)
  celeritas_target_link_libraries(celeritas PRIVATE OpenMP::OpenMP_CXX)
else()
  celeritas_target_compile_options(celeritas PRIVATE -Wno-unknown-pragmas)
endif()

celeritas_target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

celeritas_target_include_directories(celeritas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CELERITAS_USE_HIP)
  # TODO: host code includes RNG; super hacky workaround
  target_include_directories(celeritas
    SYSTEM PUBLIC
      "$ENV{ROCM_PATH}/hiprand/include"
      "$ENV{ROCM_PATH}/rocrand/include"
      "$ENV{ROCM_PATH}/hip/include"
  )
endif()

celeritas_install(TARGETS celeritas
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/celeritas_version.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/"
  COMPONENT development
  FILES_MATCHING PATTERN "*.hh"
  # TODO: ignore .cuda.hh when cuda disabled; etc
)

#----------------------------------------------------------------------------#
# GEANT/ROOT I/O MODULE
#----------------------------------------------------------------------------#

set(_IO_LIBTYPE) # empty for automatic
set(_IO_SOURCES)
set(_IO_PRIVATE_DEPS celeritas)

if(CELERITAS_USE_ROOT)
  include(${ROOT_USE_FILE})

  root_generate_dictionary(CeleritasRootInterface
    io/ImportElement.hh
    io/ImportMaterial.hh
    io/ImportParticle.hh
    io/ImportPhysicsTable.hh
    io/ImportPhysicsVector.hh
    io/ImportProcess.hh
    io/ImportVolume.hh
    io/ImportData.hh
    MODULE celeritas_io
    LINKDEF io/RootInterfaceLinkDef.h
  )
  list(APPEND _IO_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx
    io/RootExporter.cc
    io/RootImporter.cc
    io/detail/TFileUniquePtr.root.cc
  )
  # Note that ROOT requires *shared* libraries due to runtime initialization,
  # *and* it must be forcibly linked into any ROOT-using application by ensuring
  # `-Wl,--no-as-needed` for executables (on Ubuntu, which defaults to linking
  # only as needed).
  set(_IO_LIBTYPE SHARED)
  list(APPEND _IO_PRIVATE_DEPS
    ROOT::Core ROOT::Tree
  )
endif()

if(CELERITAS_USE_Geant4)
  list(APPEND _IO_SOURCES
    io/GeantImporter.cc
    io/GeantSetup.cc
    io/detail/ActionInitialization.cc
    io/detail/DetectorConstruction.cc
    io/detail/GeantBremsstrahlungProcess.cc
    io/detail/GeantExceptionHandler.cc
    io/detail/GeantLoggerAdapter.cc
    io/detail/GeantPhysicsList.cc
    io/detail/ImportProcessConverter.cc
    io/detail/PrimaryGeneratorAction.cc
  )
  list(APPEND _IO_PRIVATE_DEPS
    XercesC::XercesC
    ${Geant4_LIBRARIES}
  )
endif()

if(NOT CELERITAS_USE_Geant4 AND NOT CELERITAS_USE_ROOT)
  # No sources to compile: header only
  set(_IO_LIBTYPE INTERFACE)
  set(_IO_PRIVATE_DEPS)
endif()

add_library(celeritas_io ${_IO_LIBTYPE} ${_IO_SOURCES})
if(_IO_PRIVATE_DEPS)
  celeritas_target_link_libraries(celeritas_io
    PRIVATE ${_IO_PRIVATE_DEPS}
  )
endif()
add_library(Celeritas::IO ALIAS celeritas_io)

#---------------------------------------------------------------------------##
