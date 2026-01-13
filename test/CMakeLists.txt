#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#
# TEST HARNESS
#-----------------------------------------------------------------------------#

file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" CELERITAS_SOURCE_DIR)
configure_file(celeritas_test_config.h.in celeritas_test_config.h @ONLY)

set(_harness_sources
  AssertionHelper.cc
  Test.cc
  testdetail/JsonComparer.cc
  testdetail/NonMasterResultPrinter.cc
  testdetail/TestMacrosImpl.cc
  testdetail/TestMainImpl.cc
)
celeritas_add_test_library(testcel_harness ${_harness_sources})
target_compile_features(testcel_harness PUBLIC cxx_std_17)
celeritas_target_link_libraries(testcel_harness
  PUBLIC Celeritas::corecel GTest::gtest
  PRIVATE nlohmann_json::nlohmann_json Celeritas::ExtMPI
)
celeritas_target_include_directories(testcel_harness
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  # GCC 13 and 14 have false positives with array bounds checking, string bounds
  # checking, initialization checking, ...
  celeritas_target_compile_options(testcel_harness
    PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-array-bounds;-Wno-stringop-overflow;-Wno-uninitialized>"
  )
endif()

#-----------------------------------------------------------------------------#
# TEST OPTIONS
#-----------------------------------------------------------------------------#

function(celeritas_add_device_test base)
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    set(_cuda_args GPU SOURCES "${base}.test.cu")
  endif()
  celeritas_add_test("${base}.test.cc" ${_cuda_args} ${ARGN})
endfunction()

if(CELERITAS_UNITS STREQUAL "CGS")
  set(_fixme_cgs)
else()
  # Unit tests haven't yet been updated to include the correct units
  set(_fixme_cgs DISABLE)
endif()

# Set up Geant4 environment variables for testing
if(CELERITAS_USE_Geant4)
  # Set environment variables from Geant4-exported configuration or
  # environment currently used only for CTest setup
  celeritas_get_g4env(CELER_G4ENV)
endif()

#-----------------------------------------------------------------------------#

add_subdirectory(corecel)
add_subdirectory(geocel)
add_subdirectory(orange)
add_subdirectory(celeritas)

if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
endif()

if(CELERITAS_USE_LArSoft)
  add_subdirectory(larceler)
endif()

celeritas_setup_tests(SERIAL PREFIX testdetail)
celeritas_add_test(TestMacros.test.cc)
celeritas_add_test(JsonComparer.test.cc)

#-----------------------------------------------------------------------------#
