#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

include(CeleritasGenInteractor)

#----------------------------------------------------------------------------#
# CONFIGURE FILE
#----------------------------------------------------------------------------#
set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

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

# Generate model launcher code for host/device
set(_gen_sources)
celeritas_gen_interactor(_gen_sources "BetheHeitler" "bethe_heitler")
celeritas_gen_interactor(_gen_sources "CombinedBrem" "combined_brem")
celeritas_gen_interactor(_gen_sources "EPlusGG" "eplusgg")
celeritas_gen_interactor(_gen_sources "KleinNishina" "klein_nishina")
celeritas_gen_interactor(_gen_sources "LivermorePE" "livermore_pe")
celeritas_gen_interactor(_gen_sources "MollerBhabha" "moller_bhabha")
celeritas_gen_interactor(_gen_sources "MuBremsstrahlung" "mu_bremsstrahlung")
celeritas_gen_interactor(_gen_sources "Rayleigh" "rayleigh")
celeritas_gen_interactor(_gen_sources "RelativisticBrem" "relativistic_brem")
celeritas_gen_interactor(_gen_sources "SeltzerBerger" "seltzer_berger")
list(APPEND SOURCES ${_gen_sources})

# Main library
list(APPEND SOURCES
  base/Assert.cc
  base/ColorUtils.cc
  base/Copier.cc
  base/DeviceAllocation.cc
  comm/KernelDiagnostics.cc
  base/ScopedStreamRedirect.cc
  base/TypeDemangler.cc
  base/VectorUtils.cc
  base/detail/ReprImpl.cc
  comm/Communicator.cc
  comm/Device.cc
  comm/Logger.cc
  comm/LoggerTypes.cc
  comm/ScopedMpiInit.cc
  comm/detail/LoggerMessage.cc
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
  physics/em/RayleighModel.cc
  physics/em/RayleighProcess.cc
  physics/em/RelativisticBremModel.cc
  physics/em/SeltzerBergerModel.cc
  physics/em/detail/Utils.cc
  physics/grid/ValueGridBuilder.cc
  physics/grid/ValueGridInserter.cc
  physics/grid/ValueGridInterface.cc
  physics/material/MaterialParams.cc
  physics/material/detail/Utils.cc
  random/detail/RngStateInit.cc
  sim/TrackInitParams.cc
  sim/detail/InitializeTracks.cc
)

if(CELERITAS_USE_CUDA)
  list(APPEND SOURCES
    base/KernelParamCalculator.cuda.cc
    base/detail/Filler.cu
    random/detail/RngStateInit.cu
    sim/detail/InitializeTracks.cu
  )
  # TODO: CUDA RNG is included by host code
  list(APPEND PUBLIC_DEPS CUDA::cudart)
else()
  list(APPEND SOURCES
    random/detail/curand.nocuda.cc
    random/detail/RngStateInit.nocuda.cc
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
    geometry/GeoMaterialParams.cc
    vecgeom/VecgeomParams.cc
    vecgeom/detail/ScopedTimeAndRedirect.cc
    vecgeom/detail/VecgeomNavCollection.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml)
  # This needs to be public because its might be needed
  # to resolve the symbol generate by the `nvcc -dlink` of
  # one of the executable.
  list(APPEND PUBLIC_DEPS VecGeom::vecgeom)
endif()

celeritas_add_library(celeritas ${SOURCES})
celeritas_strip_alias(celeritas_target celeritas)
add_library(Celeritas::Core ALIAS ${celeritas_target})

if(CELERITAS_USE_OpenMP)
  find_package(OpenMP)
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

celeritas_install(TARGETS celeritas
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/celeritas_version.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# TODO: ignore .cuda.hh when cuda disabled; etc
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/"
  COMPONENT development
  FILES_MATCHING PATTERN "*.hh"
  # TODO: ignore .cuda.hh when cuda disabled; etc
)

#----------------------------------------------------------------------------#
# ROOT I/O MODULE
#----------------------------------------------------------------------------#

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
    io/RootImporter.hh
    MODULE celeritas_root
    LINKDEF io/RootInterfaceLinkDef.h
  )

  # Note that ROOT requires *shared* libraries due to runtime initialization,
  # *and* it must be forcibly linked into any ROOT-using application by ensuring
  # `-Wl,--no-as-needed` for executables (on Ubuntu, which defaults to linking
  # only as needed).
  add_library(celeritas_root SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/CeleritasRootInterface.cxx
    io/RootImporter.cc
  )
  celeritas_target_link_libraries(celeritas_root
    PRIVATE celeritas ROOT::Core ROOT::Tree
  )
else()
  # ROOT is disabled: add an importer that
  add_library(celeritas_root
    io/RootImporter.noroot.cc
  )
  celeritas_target_link_libraries(celeritas_root PRIVATE celeritas)
endif()

add_library(Celeritas::ROOT ALIAS celeritas_root)

#---------------------------------------------------------------------------##
