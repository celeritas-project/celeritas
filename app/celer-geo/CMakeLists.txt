#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  celer-geo.cc
)
set(_rasterizer_libs
  Celeritas::corecel
  Celeritas::geocel
  Celeritas::orange
  nlohmann_json::nlohmann_json
)
if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  list(APPEND _rasterizer_libs Celeritas::orange)
elseif(CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND _rasterizer_libs VecGeom::vecgeom)
elseif(CELERITAS_CORE_GEO STREQUAL "Geant4")
  celeritas_get_g4libs(_g4_geo_libs geometry)
  list(APPEND _rasterizer_libs ${_g4_geo_libs})
endif()

celeritas_add_executable(celer-geo ${SOURCES})
celeritas_target_link_libraries(celer-geo ${_rasterizer_libs})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/test-harness.py")
set(_model "${CELER_APP_DATA_DIR}/simple-cms.gdml")
set(_env
  "CELERITAS_DEMO_EXE=$<TARGET_FILE:celer-geo>"
  "CELER_DISABLE_PARALLEL=1"
)

#-----------------------------------------------------------------------------#
function(celer_app_test ext)
  set(_labels app nomemcheck)
  set(_extra_props)
  set(_extra_env "CELER_TEST_EXT=${ext}")
  if(ext STREQUAL "gpu")
    list(APPEND _labels "gpu")
    set(_extra_props RESOURCE_LOCK gpu)
    if(CELER_DISABLE_DEVICE)
      list(APPEND _extra_props DISABLED true)
    endif()
  elseif(ext STREQUAL "cpu")
    list(APPEND _extra_env "CELER_DISABLE_DEVICE=1")
  endif()

  add_test(NAME "app/celer-geo:${ext}"
    COMMAND "${CELER_PYTHON}" "${_driver}" "${_model}"
  )

  set_tests_properties("app/celer-geo:${ext}" PROPERTIES
    ENVIRONMENT "${_env};${_extra_env}"
    REQUIRED_FILES "${_driver};${_gdml_inp}"
    LABELS "${_labels}"
    ${_props} ${_extra_props}
  )
endfunction()

#-----------------------------------------------------------------------------#

celer_app_test("gpu")
celer_app_test("cpu")

#-----------------------------------------------------------------------------#
