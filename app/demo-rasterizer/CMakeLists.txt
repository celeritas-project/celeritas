#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  demo-rasterizer.cc
  RDemoRunner.cc
  ImageIO.cc
  ImageStore.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    RDemoKernel.cu
  )
  if(CELERITAS_USE_HIP)
    set_source_files_properties(
      RDemoKernel.cu
      PROPERTIES LANGUAGE HIP
    )
  endif()
endif()

set(_rasterizer_libs
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
)
if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND _rasterizer_libs VecGeom::vecgeom)
endif()

add_executable(demo-rasterizer ${SOURCES})
celeritas_target_link_libraries(demo-rasterizer ${_rasterizer_libs})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/simple-driver.py")
set(_gdml_inp "${CELER_APP_DATA_DIR}/two-boxes.gdml")
add_test(NAME "app/demo-rasterizer"
  COMMAND "$<TARGET_FILE:Python::Interpreter>" "${_driver}" "${_gdml_inp}"
)

if((CELERITAS_CORE_GEO STREQUAL "ORANGE") OR CELER_DISABLE_DEVICE)
  # two-boxes.org.json isn't set up in the test directory
  # or device is unavailable
  set(_maybe_disable DISABLED true)
else()
  set(_maybe_disable)
endif()
set(_env
  "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-rasterizer>"
  "CELER_DISABLE_PARALLEL=1"
)
set_tests_properties("app/demo-rasterizer" PROPERTIES
  ENVIRONMENT "${_env}"
  RESOURCE_LOCK gpu
  REQUIRED_FILES "${_driver};${_gdml_inp}"
  LABELS "app;nomemcheck;gpu"
  ${_maybe_disable}
)

#-----------------------------------------------------------------------------#
