#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasAccelExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.30)

find_package(Celeritas 0.6 REQUIRED)
find_package(Geant4 REQUIRED)

if(NOT CELERITAS_USE_Geant4)
  message(SEND_ERROR "This example requires Geant4 support "
    "to be enabled in Celeritas")
endif()

add_executable(simple-offload simple-offload.cc)
cuda_rdc_target_link_libraries(simple-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.1)
  message(WARNING "Fast simulation offload requires Geant4 11.1 or higher")
endif()

add_executable(fastsim-offload fastsim-offload.cc)
cuda_rdc_target_link_libraries(fastsim-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.0)
  message(WARNING "G4VTrackingManager offload requires Geant4 11.0 or higher")
else()
  add_executable(trackingmanager-offload trackingmanager-offload.cc)
  cuda_rdc_target_link_libraries(trackingmanager-offload
    Celeritas::accel
    ${Geant4_LIBRARIES}
  )
endif()

# END EXAMPLE CODE

#-----------------------------------------------------------------------------#
# Add tests for CI
#-----------------------------------------------------------------------------#

include(CTest)
set(_g4data_env)
foreach(_ds IN LISTS Geant4_DATASETS)
  list(APPEND _g4data_env
    "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
endforeach()

set(_mpi_env)
if(CELERITAS_USE_MPI)
  set(_mpi_env "CELER_DISABLE_PARALLEL=1")
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  if("$ENV{CELER_DISABLE_DEVICE}")
    message(WARNING
      "Disabling GPU testing because CELER_DISABLE_DEVICE is set"
    )
    set(CELER_DISABLE_DEVICE TRUE)
  else()
    set(CELER_DISABLE_DEVICE FALSE)
  endif()
else()
  # No device support
  set(CELER_DISABLE_DEVICE TRUE)
endif()

# Get correct capitalization for run manager type
set(_rm_serial Serial)
set(_rm_mt MT)
set(_rm_task Tasking)

# Set whether run manager is available
set(_rm_avail_serial TRUE)
set(_rm_avail_mt TRUE)
set(_rm_avail_task TRUE)
if(NOT Geant4_multithreaded_FOUND)
  set(_rm_avail_mt FALSE)
endif()
if(Geant4_VERSION VERSION_LESS 11.0)
  set(_rm_avail_task FALSE)
endif()

# Add all permutations of an example
function(add_example target)
  cmake_parse_arguments(PARSE "DISABLE" "" "" ${ARGN})
  foreach(_offload gpu cpu g4 ko)
    foreach(_rmtype serial mt task)
      set(_disable ${PARSE_DISABLE})
      if(_offload STREQUAL "gpu" AND CELER_DISABLE_DEVICE)
        set(_disable TRUE)
      endif()
      if(NOT _rm_avail_${_rmtype})
        set(_disable TRUE)
      endif()
      add_example_impl(${target} ${_offload} ${_rmtype} ${_disable})
    endforeach()
  endforeach()
endfunction()

# Add a single test
function(add_example_impl target offload rmtype disabled)
  set(_test_name "${target}:${offload}:${rmtype}")
  add_test(NAME "${_test_name}" COMMAND "$<TARGET_FILE:${target}>")
  set(_env
    ${_g4data_env}
    ${_mpi_env}
    "G4RUN_MANAGER_TYPE=${_rm_${rmtype}}"
  )
  set(_labels "app")
  if(offload STREQUAL "gpu")
    set(_extra_props RESOURCE_LOCK gpu)
    list(APPEND _labels "gpu")
  elseif(offload STREQUAL "cpu")
    list(APPEND _env "CELER_DISABLE_DEVICE=1")
  elseif(offload STREQUAL "g4")
    list(APPEND _env "CELER_DISABLE=1")
  elseif(offload STREQUAL "ko")
    list(APPEND _env "CELER_KILL_OFFLOAD=1")
  endif()
  if(NOT rmtype STREQUAL "serial")
    list(APPEND _env "G4FORCENUMBEROFTHREADS=2")
    list(APPEND _extra_props PROCESSORS 2)
  endif()

  set_tests_properties("${_test_name}" PROPERTIES
    ENVIRONMENT "${_env}"
    LABELS "${_labels}"
    DISABLED "${disabled}"
    ${_extra_props}
    ${ARGN}
  )
endfunction()

#-----------------------------------------------------------------------------#

set(_needs_g411)
if(Geant4_VERSION VERSION_LESS 11.1)
  set(_needs_g411 DISABLE)
endif()

add_example(simple-offload)
add_example(fastsim-offload ${_needs_g411})

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  # Requires 11.0 to build
  add_example(trackingmanager-offload)
endif()
