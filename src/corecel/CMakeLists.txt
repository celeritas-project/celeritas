#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  Assert.cc
  cont/Label.cc
  data/Copier.cc
  data/DeviceAllocation.cc
  io/BuildOutput.cc
  io/ColorUtils.cc
  io/ExceptionOutput.cc
  io/Logger.cc
  io/LoggerTypes.cc
  io/OutputInterface.cc
  io/OutputManager.cc
  io/ScopedStreamRedirect.cc
  io/ScopedTimeAndRedirect.cc
  io/StringUtils.cc
  io/detail/LoggerMessage.cc
  io/detail/ReprImpl.cc
  math/VectorUtils.cc
  sys/Device.cc
  sys/Environment.cc
  sys/KernelRegistry.cc
  sys/MpiCommunicator.cc
  sys/MultiExceptionHandler.cc
  sys/ScopedMpiInit.cc
  sys/ScopedSignalHandler.cc
  sys/TypeDemangler.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent code/dependencies
#-----------------------------------------------------------------------------#

list(APPEND PRIVATE_DEPS Celeritas::DeviceToolkit)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    data/detail/Filler.cu
    sys/KernelParamCalculator.device.cc
  )
endif()

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    sys/DeviceIO.json.cc
    sys/KernelRegistryIO.json.cc
  )
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_MPI)
  list(APPEND PUBLIC_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(corecel ${SOURCES})

# Require at least C++14
target_compile_features(corecel PUBLIC cxx_std_14)

celeritas_target_include_directories(corecel
  PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CELERITAS_HEADER_CONFIG_DIRECTORY}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
celeritas_target_link_libraries(corecel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
