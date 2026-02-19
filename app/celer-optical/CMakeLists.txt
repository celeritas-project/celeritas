#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  celer-optical.cc
)
set(LIBRARIES
  Celeritas::celeritas
  Celeritas::ExtDeviceApi
  Celeritas::ExtOpenMP
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  celer_app_utils
)

# Add the executable
celeritas_add_executable(celer-optical ${SOURCES})
celeritas_target_link_libraries(celer-optical PRIVATE ${LIBRARIES})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

#-----------------------------------------------------------------------------#
function(celer_optical_add_test ext gdml backend)
  set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/driver.py")
  set(_gdml_inp "${CELER_APP_DATA_DIR}/${gdml}")
  set(_env
    ${CELER_PYTHONENV}
    ${CELER_G4ENV}
    ${CELER_OMP_ENV}
    "CELERITAS_EXE=$<TARGET_FILE:celer-optical>"
    "CELER_DISABLE_PARALLEL=1"
    "CELER_CORE_GEO=${CELERITAS_CORE_GEO}"
  )
  set(_labels app nomemcheck)

  if(NOT CELERITAS_USE_Geant4 OR NOT CELERITAS_USE_Python
     OR CELERITAS_REAL_TYPE STREQUAL "float"
     OR CELERITAS_CORE_GEO STREQUAL "Geant4"
     OR (NOT VecGeom_GDML_FOUND AND CELERITAS_CORE_GEO STREQUAL "VecGeom"))
    list(APPEND _props DISABLED TRUE)
  endif()

  if("${backend}" STREQUAL "cpu")
    list(APPEND _env "CELER_DISABLE_DEVICE=1")
  elseif("${backend}" STREQUAL "gpu")
    if(CELER_DISABLE_DEVICE)
      list(APPEND _props DISABLED TRUE)
    else()
      list(APPEND _props RESOURCE_LOCK gpu)
      list(APPEND _labels gpu)
    endif()
  else()
    message(SEND_ERROR "Invalid backend type ${backend}")
  endif()

  set(_name "app/celer-optical:${ext}:${backend}")
  add_test(NAME "${_name}"
    COMMAND "${CELER_PYTHON}" "${_driver}" "${_gdml_inp}"
  )
  set_tests_properties("${_name}" PROPERTIES
    ENVIRONMENT "${_env}"
    REQUIRED_FILES "${_driver};${_gdml_inp}"
    LABELS "${_labels}"
    ${CELER_NEEDS_PYTHON}
    ${_props}
  )
endfunction()

#-----------------------------------------------------------------------------#

celer_optical_add_test(lar lar-sphere.gdml cpu)
celer_optical_add_test(lar lar-sphere.gdml gpu)

#-----------------------------------------------------------------------------#
