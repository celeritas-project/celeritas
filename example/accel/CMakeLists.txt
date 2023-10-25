#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18)
project(CeleritasAccelExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.22)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/..")
find_package(Celeritas 0.4 REQUIRED)
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

# END EXAMPLE CODE

#-----------------------------------------------------------------------------#
# Add tests for CI
#-----------------------------------------------------------------------------#

include(CTest)
function(add_example target)
  add_test(NAME "${target}" COMMAND "$<TARGET_FILE:${target}>")
endfunction()

set(_test_env)
foreach(_ds IN LISTS Geant4_DATASETS)
  list(APPEND _test_env
    "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
endforeach()

#-----------------------------------------------------------------------------#

add_example(simple-offload)
add_example(fastsim-offload)

set_tests_properties(simple-offload fastsim-offload PROPERTIES
  ENVIRONMENT "${_test_env}"
  LABELS "app"
)

if(Geant4_VERSION VERSION_LESS 11.1)
  set_tests_properties(fastsim-offload PROPERTIES
    DISABLED true
  )
endif()

