#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#
# TEST HARNESS
#-----------------------------------------------------------------------------#

file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" CELERITAS_SOURCE_DIR)
configure_file(celeritas_test_config.h.in celeritas_test_config.h @ONLY)

set(_harness_sources
  Test.cc
  testdetail/NonMasterResultPrinter.cc
  testdetail/TestMacrosImpl.cc
  testdetail/TestMainImpl.cc
)
if(CELERITAS_USE_JSON)
  list(APPEND _harness_sources
    testdetail/JsonComparer.json.cc
  )
endif()
celeritas_add_library(testcel_harness ${_harness_sources})
target_compile_features(testcel_harness PUBLIC cxx_std_17)
celeritas_target_link_libraries(testcel_harness
  PUBLIC Celeritas::corecel GTest::GTest
  PRIVATE ${nlohmann_json_LIBRARIES}
)
celeritas_target_include_directories(testcel_harness
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

#-----------------------------------------------------------------------------#
# TEST OPTIONS
#-----------------------------------------------------------------------------#

function(celeritas_add_device_test base)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    set(_cuda_args GPU SOURCES "${base}.test.cu")
  endif()
  celeritas_add_test("${base}.test.cc" ${_cuda_args} ${ARGN})
endfunction()

if(CELERITAS_USE_Geant4)
  # Optional dependence on low-energy EM data
  set(CELERITASTEST_G4ENV)
  foreach(_ds G4EMLOW G4ENSDFSTATE)
    list(APPEND CELERITASTEST_G4ENV
      "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}"
    )
  endforeach()
endif()

if(CELERITAS_UNITS STREQUAL "CGS")
  set(_fixme_cgs)
else()
  # Unit tests haven't yet been updated to include the correct units
  set(_fixme_cgs DISABLE)
endif()

#-----------------------------------------------------------------------------#

add_subdirectory(corecel)
add_subdirectory(geocel)
add_subdirectory(orange)
add_subdirectory(celeritas)

if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
endif()

celeritas_setup_tests(SERIAL PREFIX testdetail)
celeritas_add_test(TestMacros.test.cc)
celeritas_add_test(JsonComparer.test.cc)

#-----------------------------------------------------------------------------#
