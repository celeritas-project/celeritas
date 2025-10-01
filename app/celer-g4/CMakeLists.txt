#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4)
  set(SOURCES
    celer-g4.cc
    ActionInitialization.cc
    DetectorConstruction.cc
    EventAction.cc
    GeantDiagnostics.cc
    GlobalSetup.cc
    HepMC3PrimaryGeneratorAction.cc
    LogHandlers.cc
    RunAction.cc
    RunInput.cc
    RunInputIO.json.cc
    SensitiveDetector.cc
    SensitiveHit.cc
    TrackingAction.cc
  )
  set(LIBRARIES
    Celeritas::accel
    Celeritas::ExtMPI
    nlohmann_json::nlohmann_json
  )
  if(CELERITAS_USE_ROOT)
    list(APPEND SOURCES
      RootIO.cc
    )
    list(APPEND LIBRARIES ROOT::Tree)
  endif()
  celeritas_get_g4libs(_g4_libs geometry persistency intercoms run tasking
    physicslists interfaces)
  list(APPEND LIBRARIES ${_g4_libs})
else()
  set(SOURCES
    celer-g4.nogeant.cc
  )
  set(LIBRARIES Celeritas::corecel)
endif()

celeritas_add_executable(celer-g4 ${SOURCES})
celeritas_target_link_libraries(celer-g4
  ${LIBRARIES}
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/test-harness.py")
set(_exe "$<TARGET_FILE:celer-g4>")
set(_model "${CELER_APP_DATA_DIR}/simple-cms.gdml")
if(NOT CELERITAS_DEBUG)
  set(_events
    "${CELER_APP_DATA_DIR}/ttbarsplit-12evt-1108prim.hepmc3")
else()
  # Use fewer events for debug
  set(_events
    "${CELER_APP_DATA_DIR}/ttbarbits-12evt-118prim.hepmc3")
endif()

set(_env
  ${CELER_PYTHONENV}
  ${CELER_G4ENV}
  "CELER_LOG_LOCAL=debug"
  "CELER_DISABLE_PARALLEL=1"
  "CELER_DEBUG=${CELERITAS_DEBUG}"
  "CELER_USE_ROOT=${CELERITAS_USE_ROOT}"
  "CELER_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
)
set(_required_files "${_driver}" "${_exe}" "${_model}" "${_events}")
set(_props ${CELER_NEEDS_PYTHON})
if(Geant4_multithreaded_FOUND)
  list(APPEND _env "G4FORCENUMBEROFTHREADS=2")
  list(APPEND _props PROCESSORS 2)
endif()
if(CELERITAS_USE_OpenMP)
  list(APPEND _env "OMP_NUM_THREADS=1")
endif()

if(NOT CELERITAS_USE_HepMC3 OR NOT CELERITAS_USE_Geant4)
  set(_props DISABLED true)
endif()
if(CELERITAS_CORE_GEO STREQUAL "Geant4")
  # FIXME: geant4 navigator doesn't work with field navigator
  set(_props DISABLED true)
endif()
if(CELERITAS_REAL_TYPE STREQUAL "float")
  # FIXME: propagation fails in single precision
  set(_props DISABLED true)
endif()

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
  elseif(ext MATCHES "^cpu")
    list(APPEND _extra_env "CELER_DISABLE_DEVICE=1")
  else()
    list(APPEND _extra_env "CELER_DISABLE=1")
  endif()

  add_test(NAME "app/celer-g4:${ext}"
    COMMAND "${CELER_PYTHON}" "${_driver}" "${_exe}" "${_model}" "${_events}"
  )

  set_tests_properties("app/celer-g4:${ext}" PROPERTIES
    ENVIRONMENT "${_env};${_extra_env}"
    REQUIRED_FILES "${_required_files}"
    LABELS "${_labels}"
    ${_props} ${_extra_props}
  )
endfunction()

#-----------------------------------------------------------------------------#

celer_app_test("gpu")
celer_app_test("cpu")
celer_app_test("cpu-nonfatal")
celer_app_test("none")

#-----------------------------------------------------------------------------#
