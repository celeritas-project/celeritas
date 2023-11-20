#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  GCheckRunner.cc
  GCheckKernel.cc
  demo-geo-check.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND SOURCES
    GCheckKernel.cu
  )
  if(CELERITAS_USE_HIP)
    set_source_files_properties(
      GCheckKernel.cu
      PROPERTIES LANGUAGE HIP
    )
  endif()
endif()

set(LIBRARIES
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
)
if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND LIBRARIES VecGeom::vecgeom)
endif()

add_executable(demo-geo-check ${SOURCES})
celeritas_target_link_libraries(demo-geo-check ${LIBRARIES})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

set(_inp "gcheck-four-levels.json")
configure_file("${_inp}.in" "${_inp}" @ONLY)
add_test(NAME "app/demo-geo-check"
  COMMAND "$<TARGET_FILE:demo-geo-check>" "${CMAKE_CURRENT_BINARY_DIR}/${_inp}"
)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_props RESOURCE_LOCK gpu)
  set(_labels app gpu)
else()
  set(_props)
  set(_labels app)
endif()
if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  list(APPEND _props DISABLED true)
endif()
set_tests_properties("app/demo-geo-check" PROPERTIES
  ENVIRONMENT "CELER_DISABLE_PARALLEL=1"
  LABELS "${_labels}"
  ${_props}
)

#-----------------------------------------------------------------------------#
