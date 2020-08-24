#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# CONFIGURE FILE
#----------------------------------------------------------------------------#
set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
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
  comm/Utils.cc
  physics/base/ParticleParams.cc
  physics/base/ParticleStateStore.cc
  physics/base/SecondaryAllocatorStore.cc
)

if(CELERITAS_USE_CUDA)
  list(APPEND SOURCES
    base/DeviceAllocation.cuda.cc
    base/KernelParamCalculator.cuda.cc
    base/Memory.cu
    random/cuda/RngStateInit.cu
    random/cuda/RngStateStore.cc
  )
  list(APPEND PRIVATE_DEPS CUDA::cudart)
else()
  list(APPEND SOURCES
    base/DeviceAllocation.nocuda.cc
    base/Memory.nocuda.cc
  )
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
  )
  list(APPEND PRIVATE_DEPS VecGeom::vgdml VecGeom::vecgeom)
  if(CELERITAS_USE_CUDA)
    list(APPEND SOURCES
      geometry/detail/VGNavStateStore.cuda.cc
    )
    list(APPEND PRIVATE_DEPS VecGeom::vecgeomcuda)
  else()
    list(APPEND SOURCES
      geometry/detail/VGNavStateStore.nocuda.cc
    )
  endif()
endif()

add_library(celeritas ${SOURCES})
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

  root_generate_dictionary(CeleritasRootGeantParticle
    io/GeantParticle.hh
    LINKDEF io/GeantParticleLinkDef.h
  )

  # Note that ROOT requires *shared* libraries due to runtime initialization.
  # Tests that use ROOT to load the data will have to link against this library.
  add_library(CeleritasIO SHARED
    io/GeantImporter.cc
    CeleritasRootGeantParticle.cxx
  )
  add_library(Celeritas::IO ALIAS CeleritasIO)
  target_link_libraries(CeleritasIO
    PRIVATE celeritas ROOT::Core ROOT::Tree
  )
endif()

#---------------------------------------------------------------------------##
