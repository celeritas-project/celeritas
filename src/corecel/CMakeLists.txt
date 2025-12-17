#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  nlohmann_json::nlohmann_json
  Celeritas::ExtDeviceApi Celeritas::ExtThrust
)
set(PUBLIC_DEPS Celeritas::BuildFlags)

#----------------------------------------------------------------------------#
# Configure Files
#----------------------------------------------------------------------------#
# Celeritas version

celeritas_version_to_hex(CELERITAS_VERSION PROJECT_VERSION)

#----------------------------------------------------------------------------#
# Build configuration

set(CELERITAS_USE_COVFIE ${CELERITAS_USE_covfie})
set(CELERITAS_USE_GEANT4  ${CELERITAS_USE_Geant4})
set(CELERITAS_USE_HEPMC3  ${CELERITAS_USE_HepMC3})
set(CELERITAS_USE_LARSOFT ${CELERITAS_USE_LArSoft})
set(CELERITAS_USE_OPENMP  ${CELERITAS_USE_OpenMP})
set(CELERITAS_USE_PERFETTO ${CELERITAS_USE_Perfetto})
set(CELERITAS_USE_VECGEOM ${CELERITAS_USE_VecGeom})

# Set CELERITAS_CORE_RNG_CONFIG to a numeric table of RNG options and the
# selection.
# Start counter from 1 because undefined config have the implicit value of 0 in
# the C preprocessor, so any unavailable options (e.g. CELERITAS_USE_CURAND
# when HIP is in use) will implicitly be zero.
celeritas_generate_option_config(CELERITAS_CODATA)
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
foreach(_var BUILD_TYPE HOSTNAME REAL_TYPE UNITS CONSTANTS OPENMP CORE_GEO CORE_RNG)
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
foreach(_pkg CLHEP covfie CUDA HepMC3 HIP Geant4 G4VG LArSoft ROOT Thrust VecGeom)
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

# Save VecGeom options as always-on macros to simplify testing
if(DEFINED CELERITAS_VecGeom_SURFACE)
  # Override/pretend VecGeom has this option
  set(VecGeom_SURFACE ${CELERITAS_VecGeom_SURFACE})
endif()
set(CELERITAS_VECGEOM_CONFIG)
set(_vecgeom_opts)
foreach(_opt SURFACE BVH_SINGLE SINGLE_PRECISION)
  if(NOT DEFINED VecGeom_${_opt})
    set(_val "0 /* not defined */")
  elseif(VecGeom_${_opt})
    set(_val 1)
    string(TOLOWER "${_opt}" _lower_opt)
    list(APPEND _vecgeom_opts "${_lower_opt}")
  else()
    set(_val 0)
  endif()
  list(APPEND CELERITAS_VECGEOM_CONFIG
    "#define CELERITAS_VECGEOM_${_opt} ${_val}"
  )
endforeach()
foreach(_opt NAV BACKEND)
  if(DEFINED VecGeom_${_opt})
    list(APPEND _vecgeom_opts "${VecGeom_${_opt}}")
  endif()
endforeach()
if(NOT VecGeom_VECTOR STREQUAL "empty")
  list(APPEND _vecgeom_opts "${VecGeom_VECTOR}")
endif()
list(JOIN _vecgeom_opts "," _vecgeom_opts)
celeritas_append_cmake_string("vecgeom_options" "${_vecgeom_opts}")

# Save Geant4 options for verbose output: start with optional TLS model
set(_g4_opts ${Geant4_TLS_MODEL})
foreach(_opt multithreaded shared static smartstack gdml usolids)
  if(Geant4_${_opt}_FOUND)
    list(APPEND _g4_opts "${_opt}")
  endif()
endforeach()
list(JOIN _g4_opts "," _g4_opts)
celeritas_append_cmake_string("geant4_options" "${_g4_opts}")

# Convert lists to newline-separated strings
foreach(_ext CMAKE_STRINGS_DECL CMAKE_STRINGS_DEF VECGEOM_CONFIG DEPENDENCY_VERSIONS)
  list(JOIN CELERITAS_${_ext} "\n" CELERITAS_${_ext})
endforeach()

#-----------------------------------------------------------------------------#
# Generate the config files

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
  AssertIO.json.cc
  Types.cc
  data/Copier.cc
  data/DeviceAllocation.cc
  data/AuxInterface.cc
  data/AuxParamsRegistry.cc
  data/AuxStateVec.cc
  data/detail/PinnedAllocatorImpl.cc
  grid/GridTypes.cc
  grid/GridTypesIO.json.cc
  grid/SplineDerivCalculator.cc
  grid/VectorUtils.cc
  io/BuildOutput.cc
  io/ColorUtils.cc
  io/ExceptionOutput.cc
  io/JsonUtils.json.cc
  io/Label.cc
  io/LogContextException.cc
  io/LogHandlers.cc
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
  random/data/CuHipRngData.cc
  random/data/XorwowRngData.cc
  random/data/RanluxppRngData.cc
  random/distribution/DistributionInserter.cc
  random/params/CuHipRngParams.cc
  random/params/RanluxppRngParams.cc
  random/params/XorwowRngParams.cc
  sys/ActionInterface.cc
  sys/ActionRegistry.cc
  sys/ActionRegistryOutput.cc
  sys/DeviceIO.json.cc
  sys/Environment.cc
  sys/KernelRegistry.cc
  sys/KernelRegistryIO.json.cc
  sys/MemRegistry.cc
  sys/MemRegistryIO.json.cc
  sys/ScopedMem.cc
  sys/ScopedSignalHandler.cc
  sys/TypeDemangler.cc
  sys/Version.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent object libraries
#-----------------------------------------------------------------------------#

# All these classes depend on OpenMP if available
celeritas_add_object_library(corecel_openmp_obj
  sys/MultiExceptionHandler.cc
)
celeritas_target_link_libraries(corecel_openmp_obj
  PRIVATE Celeritas::ExtOpenMP Celeritas::BuildFlags
)
list(APPEND SOURCES $<TARGET_OBJECTS:corecel_openmp_obj>)
list(APPEND PRIVATE_DEPS corecel_openmp_obj)

#-----------------------------------------------------------------------------#

# All these classes depend on MPI if available
celeritas_add_object_library(corecel_mpi_obj
  Assert.cc
  io/Logger.cc
  sys/Device.cc
  sys/MpiCommunicator.cc
  sys/ScopedMpiInit.cc
)
celeritas_target_link_libraries(corecel_mpi_obj
  PRIVATE Celeritas::ExtMPI Celeritas::ExtDeviceApi Celeritas::BuildFlags
)
list(APPEND SOURCES $<TARGET_OBJECTS:corecel_mpi_obj>)
list(APPEND PRIVATE_DEPS corecel_mpi_obj)

#-----------------------------------------------------------------------------#

set(_profiling_sources
  sys/ScopedProfiling.cc
)
if(CELERITAS_USE_Perfetto)
  list(APPEND _profiling_sources
    sys/detail/TraceCounterImpl.perfetto.cc
    sys/TracingSession.perfetto.cc
  )
endif()
celeritas_add_object_library(corecel_profiling_obj ${_profiling_sources})
celeritas_target_link_libraries(corecel_profiling_obj
  PRIVATE Celeritas::ExtPerfetto Celeritas::BuildFlags
)
list(APPEND SOURCES $<TARGET_OBJECTS:corecel_profiling_obj>)
list(APPEND PRIVATE_DEPS corecel_profiling_obj)

#-----------------------------------------------------------------------------#
# CUDA/HIP code
#-----------------------------------------------------------------------------#

macro(celeritas_polysource filename_we)
  celeritas_polysource_append(SOURCES ${filename_we})
endmacro()

celeritas_polysource(random/data/detail/CuHipRngStateInit)
celeritas_polysource(random/data/detail/RanluxppRngStateInit)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    data/Filler.cu
    sys/KernelParamCalculator.device.cc
    sys/Stream.cu
    sys/detail/AsyncMemoryResource.cu
  )
endif()

if(CELERITAS_USE_CUDA)
  # Note: we currently assume the CUDA install includes the nvtx header
  list(APPEND SOURCES
    sys/detail/NvtxUtils.cc
    sys/detail/TraceCounterImpl.cuda.cc
    sys/ScopedLimitSaver.cuda.cc
    sys/ScopedProfiling.cuda.cc
  )
endif()

if(CELERITAS_USE_HIP)
  list(APPEND SOURCES
    sys/ScopedProfiling.hip.cc
  )
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(corecel ${SOURCES})
celeritas_target_link_libraries(corecel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
