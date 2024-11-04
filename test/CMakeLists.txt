#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

include(CeleritasAddTest)

#-----------------------------------------------------------------------------#

function(celeritas_add_test_library target)
  cuda_rdc_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_rdc_sources_contains_cuda(_cuda_sources ${_sources})
  if(CELERITAS_USE_HIP AND _cuda_sources)
    # When building Celeritas libraries, we put HIP/CUDA files in shared .cu
    # suffixed files. Override the language if using HIP.
    set_source_files_properties(
      ${_cuda_sources}
      PROPERTIES LANGUAGE HIP
    )
  endif()

  celeritas_cuda_rdc_wrapper_add_library(${target} ${ARGN})

  # We could hide this behind `if (CELERITAS_USE_ROOT)`
  get_target_property(_tgt ${target} CUDA_RDC_OBJECT_LIBRARY)
  if(_tgt)
    set_target_properties(${_tgt} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
    )
  endif()
endfunction()

#-----------------------------------------------------------------------------#
# TEST HARNESS
#-----------------------------------------------------------------------------#

file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" CELERITAS_SOURCE_DIR)
configure_file(celeritas_test_config.h.in celeritas_test_config.h @ONLY)

if(CELERITAS_USE_MPI)
  set(_mpi_libs MPI::MPI_CXX)
else()
  set(_mpi_libs)
endif()

set(_harness_sources
  Test.cc
  testdetail/JsonComparer.json.cc
  testdetail/NonMasterResultPrinter.cc
  testdetail/TestMacrosImpl.cc
  testdetail/TestMainImpl.cc
)
celeritas_add_test_library(testcel_harness ${_harness_sources})
target_compile_features(testcel_harness PUBLIC cxx_std_17)
celeritas_target_link_libraries(testcel_harness
  PUBLIC Celeritas::corecel GTest::gtest
  PRIVATE nlohmann_json::nlohmann_json ${_mpi_libs}
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
