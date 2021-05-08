#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

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

# Main library
list(APPEND SOURCES
  base/Assert.cc
  base/ColorUtils.cc
  base/DeviceAllocation.cc
  comm/KernelDiagnostics.cc
  base/ScopedStreamRedirect.cc
  base/TypeDemangler.cc
  base/detail/Copier.cc
  comm/Communicator.cc
  comm/Device.cc
  comm/Logger.cc
  comm/LoggerTypes.cc
  comm/ScopedMpiInit.cc
  comm/detail/LoggerMessage.cc
  geometry/detail/ScopedTimeAndRedirect.cc
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
  physics/em/ComptonProcess.cc
  physics/em/PhotoelectricProcess.cc
  physics/em/LivermorePEModel.cc
  physics/em/EIonizationProcess.cc
  physics/em/EPlusAnnihilationProcess.cc
  physics/em/EPlusGGModel.cc
  physics/em/GammaConversionProcess.cc
  physics/em/KleinNishinaModel.cc
  physics/em/MollerBhabhaModel.cc
  physics/em/RayleighModel.cc
  physics/em/RayleighProcess.cc
  physics/em/detail/Rayleigh.cc
  physics/em/SeltzerBergerModel.cc
  physics/em/detail/Utils.cc
  physics/grid/ValueGridBuilder.cc
  physics/grid/ValueGridInserter.cc
  physics/grid/ValueGridInterface.cc
  physics/material/MaterialParams.cc
  physics/material/detail/Utils.cc
  random/RngInterface.cc
)

if(CELERITAS_USE_CUDA)
  list(APPEND SOURCES
    base/KernelParamCalculator.cuda.cc
    base/detail/Filler.cu
    physics/em/detail/BetheHeitler.cu
    physics/em/detail/EPlusGG.cu
    physics/em/detail/KleinNishina.cu
    physics/em/detail/MollerBhabha.cu
    physics/em/detail/LivermorePE.cu
    physics/em/detail/Rayleigh.cu
    random/detail/RngStateInit.cu
    sim/detail/SimStateInit.cu
  )
  # TODO: CUDA RNG is included by host code
  list(APPEND PUBLIC_DEPS CUDA::cudart)
else()
  list(APPEND SOURCES
    random/detail/curand.nocuda.cc
    random/detail/RngStateInit.nocuda.cc
    sim/detail/SimStateInit.nocuda.cc
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
  )
  list(APPEND PUBLIC_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_MPI)
  list(APPEND PUBLIC_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    geometry/GeoMaterialParams.cc
    geometry/GeoParams.cc
    geometry/detail/VGNavCollection.cc
    sim/ParamStore.cc
    sim/StateStore.cc
    sim/TrackInitializerStore.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml VecGeom::vecgeom)
  if(CELERITAS_USE_CUDA)
    list(APPEND SOURCES
      sim/detail/InitializeTracks.cu
    )
  else()
    list(APPEND SOURCES
      sim/detail/InitializeTracks.nocuda.cc
    )
  endif()
endif()

celeritas_add_library(celeritas ${SOURCES} SHARED)
add_library(Celeritas::Core ALIAS celeritas)

target_link_libraries(celeritas
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

target_include_directories(celeritas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
  celeritas_link_vecgeom_cuda(celeritas)
endif()

install(TARGETS celeritas
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/celeritas_version.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

foreach(_SUBDIR base comm geometry io physics random sim
    base/detail comm/detail geometry/detail sim/detail)
  file(GLOB _HEADERS
    "${_SUBDIR}/*.hh"
  )
  install(FILES ${_HEADERS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${_SUBDIR}/"
  )
endforeach()

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
  target_link_libraries(celeritas_root
    PRIVATE celeritas ROOT::Core ROOT::Tree
  )
else()
  # ROOT is disabled: add an importer that
  add_library(celeritas_root
    io/RootImporter.noroot.cc
  )
  target_link_libraries(celeritas_root PRIVATE celeritas)
endif()

add_library(Celeritas::ROOT ALIAS celeritas_root)

#---------------------------------------------------------------------------##
