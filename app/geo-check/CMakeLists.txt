#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(_geo_check_src
  GCheckRunner.cc
  GCheckKernel.cc
  geo-check.cc
)
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  list(APPEND _geo_check_src
    GCheckKernel.cu
  )
  if(CELERITAS_USE_HIP)
    set_source_files_properties(
      GCheckKernel.cu
      PROPERTIES LANGUAGE HIP
    )
  endif()
endif()

set(_geo_check_libs
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
)
if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND _geo_check_libs VecGeom::vecgeom)
endif()

add_executable(geo-check ${_geo_check_src})
celeritas_target_link_libraries(geo-check ${_geo_check_libs})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_BUILD_TESTS)
  return()
endif()

set(_inp "gcheck-four-levels.json")
configure_file("${_inp}.in" "${_inp}" @ONLY)
add_test(NAME "app/geo-check"
  COMMAND "$<TARGET_FILE:geo-check>" "${CMAKE_CURRENT_BINARY_DIR}/${_inp}"
)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_props
    RESOURCE_LOCK gpu
    LABELS "app;gpu"
  )
else()
  set(_props LABELS "app")
endif()
if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  list(APPEND _props DISABLED true)
endif()
set_tests_properties("app/geo-check" PROPERTIES
  ${_props}
)

#-----------------------------------------------------------------------------#
