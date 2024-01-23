#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasAccelExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")
find_package(Celeritas 0.5 REQUIRED)
find_package(Geant4 REQUIRED)

if(NOT CELERITAS_USE_Geant4 OR CELERITAS_CORE_GEO STREQUAL "ORANGE")
  message(SEND_ERROR "This example requires Geant4 support "
    "to be enabled in Celeritas")
endif()

include(CeleritasLibrary)

add_executable(simple-offload simple-offload.cc)
celeritas_target_link_libraries(simple-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.1)
  message(WARNING "Fast simulation offload requires Geant4 11.1 or higher")
endif()

add_executable(fastsim-offload fastsim-offload.cc)
celeritas_target_link_libraries(fastsim-offload
  Celeritas::accel
  ${Geant4_LIBRARIES}
)

if(Geant4_VERSION VERSION_LESS 11.0)
  message(WARNING "G4VTrackingManager offload requires Geant4 11.0 or higher")
else()
  add_executable(trackingmanager-offload trackingmanager-offload.cc)
  celeritas_target_link_libraries(trackingmanager-offload
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

function(add_example target)
  add_example_impl(${target} Serial)
  if(Geant4_multithreaded_FOUND)
    add_example_impl(${target} MT)
    add_example_impl(${target} Tasking)
  endif()
endfunction()

function(disable_example target)
  set(_tests ${target}-Serial)
  if(Geant4_multithreaded_FOUND)
    list(APPEND _tests ${target}-MT ${target}-Tasking)
  endif()
  set_tests_properties(${_tests} PROPERTIES
    DISABLED true
  )
endfunction()

function(add_example_impl target rmtype)
  set(_test_name "${target}-${rmtype}")
  add_test(NAME "${_test_name}" COMMAND "$<TARGET_FILE:${target}>")
  set_tests_properties(${_test_name} PROPERTIES
    ENVIRONMENT "${_g4data_env};G4RUN_MANAGER_TYPE=${rmtype}"
    LABELS "app"
  )
endfunction()

#-----------------------------------------------------------------------------#

add_example(simple-offload)
add_example(fastsim-offload)

if(Geant4_VERSION VERSION_LESS 11.1)
  disable_example(fastsim-offload)
endif()

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  add_example(trackingmanager-offload)
endif()
