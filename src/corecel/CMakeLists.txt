#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
  PinnedAllocator.cc
  data/Copier.cc
  data/DeviceAllocation.cc
  io/BuildOutput.cc
  io/ColorUtils.cc
  io/ExceptionOutput.cc
  io/Label.cc
  io/LogContextException.cc
  io/Logger.cc
  io/LoggerTypes.cc
  io/OutputInterface.cc
  io/OutputRegistry.cc
  io/ScopedStreamRedirect.cc
  io/ScopedTimeAndRedirect.cc
  io/StringUtils.cc
  io/detail/EnumStringMapperImpl.cc
  io/detail/LoggerMessage.cc
  io/detail/ReprImpl.cc
  sys/Device.cc
  sys/Environment.cc
  sys/KernelRegistry.cc
  sys/MemRegistry.cc
  sys/ScopedMem.cc
  sys/MpiCommunicator.cc
  sys/MultiExceptionHandler.cc
  sys/ScopedMpiInit.cc
  sys/ScopedSignalHandler.cc
  sys/Stream.cc
  sys/TypeDemangler.cc
  sys/Version.cc
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
if(CELERITAS_USE_CUDA)
  # we currently depend on CUDA install including the nvtx header
  list(APPEND SOURCES
    sys/ScopedLimitSaver.cuda.cc
    sys/ScopedProfiling.cuda.cc
  )
endif()

if(CELERITAS_USE_HIP)
  list(APPEND SOURCES
    sys/ScopedProfiling.hip.cc
  )
endif()

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    AssertIO.json.cc
    io/JsonUtils.json.cc
    sys/DeviceIO.json.cc
    sys/KernelRegistryIO.json.cc
    sys/MemRegistryIO.json.cc
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

# Require at least C++17
target_compile_features(corecel PUBLIC cxx_std_17)
if(CELERITAS_USE_CUDA)
  target_compile_features(corecel PUBLIC cuda_std_17)
endif()
if(WIN32)
  target_compile_definitions(corecel PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:NOMINMAX NOGDI>
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # __cplusplus: use standard-compliant C++ version macro
    # preprocessor: use standard-compliant preprocessor
    # EHsc: enable standard C++ exception handling
    target_compile_options(corecel PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor /EHsc>
    )
  endif()
endif()


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
