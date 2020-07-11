#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# CONFIGURE FILE
#----------------------------------------------------------------------------#

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
  base/StackAllocatorStore.cc
  comm/Utils.cc
  physics/base/ParticleParams.cc
  physics/base/ParticleStateStore.cc
)

if(CELERITAS_USE_CUDA)
  list(APPEND SOURCES
    base/DeviceAllocation.cuda.cc
    base/KernelParamCalculator.cuda.cc
    random/cuda/RngStateStore.cu
    base/Memory.cu
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

add_library(celeritas ${SOURCES})
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

#---------------------------------------------------------------------------##
