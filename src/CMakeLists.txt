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

set(_CONFIG_NAME "celeritas_config.h")
configure_file("${_CONFIG_NAME}.in" "${_CONFIG_NAME}" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG_NAME}"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

#----------------------------------------------------------------------------#
# LIBRARY
#----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS)

list(APPEND SOURCES
  base/Assert.cc
  base/ColorUtils.cc
  base/TypeDemangler.cc
  comm/Logger.cc
  comm/LoggerTypes.cc
  comm/detail/LoggerMessage.cc
  physics/base/ParticleParams.cc
  physics/base/ParticleStateStore.cc
  physics/base/SecondaryAllocatorStore.cc
  physics/material/MaterialParams.cc
  physics/material/MaterialStateStore.cc
  physics/material/detail/Utils.cc
  random/cuda/RngStateStore.cc
  sim/SimStateStore.cc
)

if(CELERITAS_USE_CUDA)
  list(APPEND SOURCES
    base/DeviceAllocation.cuda.cc
    base/KernelParamCalculator.cuda.cc
    base/Memory.cu
    comm/Device.cuda.cc
    random/cuda/detail/RngStateInit.cu
    sim/detail/SimStateInit.cu
  )
  list(APPEND PRIVATE_DEPS CUDA::cudart)
else()
  list(APPEND SOURCES
    base/DeviceAllocation.nocuda.cc
    base/Memory.nocuda.cc
    comm/Device.nocuda.cc
    random/cuda/curand.nocuda.cc
    random/cuda/detail/RngStateInit.nocuda.cc
    sim/detail/SimStateInit.nocuda.cc
  )
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND SOURCES
    io/EventReader.cc
  )
  list(APPEND PRIVATE_DEPS ${HEPMC3_LIB})
endif()

if(CELERITAS_USE_MPI)
  list(APPEND SOURCES
    comm/Communicator.mpi.cc
    comm/ScopedMpiInit.mpi.cc
  )
  list(APPEND PUBLIC_DEPS MPI::MPI_CXX)
else()
  list(APPEND SOURCES
    comm/Communicator.nompi.cc
    comm/ScopedMpiInit.nompi.cc
  )
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    geometry/GeoParams.cc
    geometry/GeoStateStore.cc
    sim/ParamStore.cc
    sim/StateStore.cc
    sim/TrackInitializerStore.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml VecGeom::vecgeom)
  if(CELERITAS_USE_CUDA)
    list(APPEND SOURCES
      geometry/detail/VGNavStateStore.cuda.cc
      sim/detail/InitializeTracks.cu
    )
    list(APPEND PRIVATE_DEPS VecGeom::vecgeomcuda VecGeom::vecgeomcuda_static)
  else()
    list(APPEND SOURCES
      geometry/detail/VGNavStateStore.nocuda.cc
      sim/detail/InitializeTracks.nocuda.cc
    )
  endif()
endif()

add_library(celeritas ${SOURCES})
add_library(Celeritas::Core ALIAS celeritas)

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
  set_target_properties(celeritas PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

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

install(TARGETS celeritas
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

foreach(_SUBDIR base geom)
  file(GLOB _HEADERS
    "${_SUBDIR}/*.h"
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
    base/OpaqueId.hh
    io/GdmlGeometryMap.hh
    io/ImportElement.hh
    io/ImportMaterial.hh
    io/ImportParticle.hh
    io/ImportPhysicsTable.hh
    io/ImportPhysicsVector.hh
    io/ImportVolume.hh
    physics/base/ParticleMd.hh
    LINKDEF io/RootInterfaceLinkDef.h
  )

  # Note that ROOT requires *shared* libraries due to runtime initialization.
  # Tests that use ROOT to load the data will have to link against this library.
  add_library(CeleritasIO SHARED
    io/GdmlGeometryMap.cc
    io/ImportPhysicsTable.cc
    io/ImportPhysicsVector.cc
    io/RootImporter.cc
    CeleritasRootInterface.cxx
  )
  add_library(Celeritas::IO ALIAS CeleritasIO)
  target_link_libraries(CeleritasIO
    PRIVATE celeritas ROOT::Core ROOT::Tree
  )
endif()

#---------------------------------------------------------------------------##
