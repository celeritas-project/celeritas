#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS nlohmann_json::nlohmann_json)
set(PUBLIC_DEPS)

#----------------------------------------------------------------------------#
# Configure Files
#----------------------------------------------------------------------------#
# Celeritas version

celeritas_version_to_hex(CELERITAS_VERSION PROJECT_VERSION)

#----------------------------------------------------------------------------#
# Build configuration

set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})
set(CELERITAS_USE_PERFETTO ${CELERITAS_USE_Perfetto})

# Set CELERITAS_CORE_RNG_CONFIG to a numeric table of RNG options and the
# selection.
# Start counter from 1 because undefined config have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
celeritas_generate_option_config(CELERITAS_CORE_GEO)
celeritas_generate_option_config(CELERITAS_CORE_RNG)
celeritas_generate_option_config(CELERITAS_OPENMP)
celeritas_generate_option_config(CELERITAS_REAL_TYPE)
celeritas_generate_option_config(CELERITAS_UNITS)

#----------------------------------------------------------------------------#
# Detailed build configuration for reproducibility/provenance

# Get host system information
if(NOT DEFINED CELERITAS_HOSTNAME)
  set(_hostname "$ENV{LMOD_SYSTEM_NAME}")
  if(NOT _hostname)
    cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
    string(REGEX REPLACE "\\..*" "" _hostname "${_hostname}")
  endif()
  set(CELERITAS_HOSTNAME "${_hostname}" CACHE INTERNAL
    "Build/deploy system name")
endif()

# Save CMake variables as strings
set(CELERITAS_CMAKE_STRINGS)
set(CELERITAS_CMAKE_STRINGS_DECL)
set(CELERITAS_CMAKE_STRINGS_DEF)
macro(celeritas_append_cmake_string key value)
  list(APPEND CELERITAS_CMAKE_STRINGS_DECL
    "extern char const ${key}[]\;"
  )
  list(APPEND CELERITAS_CMAKE_STRINGS_DEF
    "char const ${key}[] = R\"(${value})\"\;"
  )
endmacro()

string(TOLOWER "${CMAKE_BUILD_TYPE}," CELERITAS_BUILD_TYPE)
if(BUILD_SHARED_LIBS)
  string(APPEND CELERITAS_BUILD_TYPE "shared")
else()
  string(APPEND CELERITAS_BUILD_TYPE "static")
  if(CMAKE_POSITION_INDEPENDENT_CODE)
    string(APPEND CELERITAS_BUILD_TYPE ",pic")
  endif()
endif()
foreach(_var BUILD_TYPE HOSTNAME REAL_TYPE UNITS OPENMP CORE_GEO CORE_RNG)
  string(TOLOWER "${_var}" _lower)
  celeritas_append_cmake_string("${_lower}" "${CELERITAS_${_var}}")
endforeach()

set(_gpu_arch "")
if(CELERITAS_USE_CUDA)
  set(_gpu_arch "${CMAKE_CUDA_ARCHITECTURES}")
elseif(CELERITAS_USE_HIP)
  set(_gpu_arch "${CMAKE_HIP_ARCHITECTURES}")
endif()
# Arch variable may be ;-separated list, so we need to escape the ;
string(REPLACE ";" "\;" _gpu_arch "${_gpu_arch}")
celeritas_append_cmake_string("gpu_architectures" "${_gpu_arch}")

list(APPEND CELERITAS_CMAKE_STRINGS "")

# Save dependency versions as strings
set(CUDA_VERSION_STRING "${CUDAToolkit_VERSION}")
set(HIP_VERSION_STRING "${hip-lang_VERSION}")
foreach(_pkg CLHEP CUDA HepMC3 HIP Geant4 ROOT Thrust VecGeom)
  string(TOLOWER "${_pkg}" _lower)
  if(DEFINED ${_pkg}_VERSION_STRING)
    set(_value "${${_pkg}_VERSION_STRING}")
  elseif(DEFINED ${_pkg}_VERSION)
    set(_value "${${_pkg}_VERSION}")
  else()
    set(_value "")
  endif()
  celeritas_append_cmake_string("${_lower}_version" "${_value}")
endforeach()

foreach(_ext DECL DEF)
  list(JOIN CELERITAS_CMAKE_STRINGS_${_ext} "\n" CELERITAS_CMAKE_STRINGS_${_ext})
endforeach()

# NOTE: some values are defined in top-level CMakeLists
# - CELERITAS_HAVE_ROCTX
# - CELERITAS_SINCOSPI_PREFIX

if(DEFINED CELERITAS_SINCOSPI_PREFIX)
  set(CELERITAS_SINCOSPI_PREFIX_DEF
    "#define CELERITAS_SINCOSPI_PREFIX ${CELERITAS_SINCOSPI_PREFIX}"
  )
endif()

# Save dependency versions as hex
set(CELERITAS_DEPENDENCY_VERSIONS)
foreach(_dep Geant4 VecGeom HepMC3)
  string(TOUPPER "${_dep}" _upper)
  celeritas_version_to_hex(_temp_version_hex ${_dep}_VERSION)
  list(APPEND CELERITAS_DEPENDENCY_VERSIONS
    "#define CELERITAS_${_upper}_VERSION ${_temp_version_hex}"
  )
endforeach()
list(JOIN CELERITAS_DEPENDENCY_VERSIONS "\n" CELERITAS_DEPENDENCY_VERSIONS)

# Save VecGeom surface macro
set(CELERITAS_VECGEOM_SURFACE ${VecGeom_USE_SURF})

#-----------------------------------------------------------------------------#

celeritas_configure_file("Config.hh.in" "corecel/Config.hh" @ONLY)

foreach(_file Version Config)
  celeritas_configure_file("${_file}.hh.in" "corecel/${_file}.hh" @ONLY)
  configure_file("${_file}.cc.in" "${_file}.cc" @ONLY)
  list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${_file}.cc")
endforeach()

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  Assert.cc
  AssertIO.json.cc
  Types.cc
  data/Copier.cc
  data/DeviceAllocation.cc
  data/AuxInterface.cc
  data/AuxParamsRegistry.cc
  data/AuxStateVec.cc
  data/detail/PinnedAllocatorImpl.cc
  grid/SplineDerivCalculator.cc
  grid/VectorUtils.cc
  io/BuildOutput.cc
  io/ColorUtils.cc
  io/ExceptionOutput.cc
  io/JsonUtils.json.cc
  io/Label.cc
  io/LogContextException.cc
  io/LogHandlers.cc
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
  math/TridiagonalSolver.cc
  sys/ActionInterface.cc
  sys/ActionRegistry.cc
  sys/ActionRegistryOutput.cc
  sys/Device.cc
  sys/DeviceIO.json.cc
  sys/Environment.cc
  sys/KernelRegistry.cc
  sys/KernelRegistryIO.json.cc
  sys/MemRegistry.cc
  sys/MemRegistryIO.json.cc
  sys/MpiCommunicator.cc
  sys/MultiExceptionHandler.cc
  sys/ScopedMem.cc
  sys/ScopedMpiInit.cc
  sys/ScopedProfiling.cc
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
    data/Filler.cu
    sys/KernelParamCalculator.device.cc
  )
endif()
if(CELERITAS_USE_CUDA)
  # Note: we currently assume the CUDA install includes the nvtx header
  list(APPEND SOURCES
    sys/ScopedLimitSaver.cuda.cc
    sys/ScopedProfiling.cuda.cc
  )
  # NVTX requires `-ldl`
  list(APPEND PRIVATE_DEPS ${CMAKE_DL_LIBS})
endif()

if(CELERITAS_USE_HIP)
  list(APPEND SOURCES
    sys/ScopedProfiling.hip.cc
  )
endif()

if(CELERITAS_USE_MPI)
  list(APPEND PRIVATE_DEPS MPI::MPI_CXX)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

if(CELERITAS_USE_Perfetto)
  list(APPEND PRIVATE_DEPS Celeritas::Perfetto)
  list(APPEND SOURCES
    sys/TraceCounter.perfetto.cc
    sys/TracingSession.cc
  )
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(corecel ${SOURCES})

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
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  # GCC 13 and 14 have *lots* of false positives, and not just on our project
  # see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107532
  celeritas_target_compile_options(corecel
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>"
  )
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
