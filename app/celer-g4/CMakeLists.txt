#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
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
    PGPrimaryGeneratorAction.cc
    RunAction.cc
    RunInput.cc
    SensitiveDetector.cc
    SensitiveHit.cc
    TimerOutput.cc
    TrackingAction.cc
  )
  set(LIBRARIES Celeritas::accel)
  if(CELERITAS_USE_JSON)
    list(APPEND SOURCES
      RunInputIO.json.cc
    )
    list(APPEND LIBRARIES
      nlohmann_json::nlohmann_json
    )
  endif()
else()
  set(SOURCES
    celer-g4.nogeant.cc
  )
  set(LIBRARIES)
endif()

if(CELERITAS_USE_ROOT AND CELERITAS_USE_Geant4)
  list(APPEND SOURCES
    RootIO.cc
  )
  list(APPEND LIBRARIES ROOT::Tree)
endif()

celeritas_add_executable(celer-g4 ${SOURCES})
celeritas_target_link_libraries(celer-g4
  ${LIBRARIES}
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_BUILD_TESTS)
  return()
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CELERG4_MAX_NUM_TRACKS 524288)
    set(CELERG4_INIT_CAPACITY 4194304)
  else()
    # Use smaller number of tracks when running on CPU
    set(CELERG4_MAX_NUM_TRACKS 2048)
    set(CELERG4_INIT_CAPACITY 65536)
  endif()
else()
  # Use smaller number of tracks when running on CPU
  set(CELERG4_MAX_NUM_TRACKS 1024)
  set(CELERG4_INIT_CAPACITY 32768)
endif()

if(NOT CELERITAS_DEBUG)
  set(CELERG4_EVENTS
    "${CELER_APP_DATA_DIR}/ttbarsplit-12evt-1108prim.hepmc3")
else()
  # Use fewer events for debug
  set(CELERG4_EVENTS
    "${CELER_APP_DATA_DIR}/ttbarbits-12evt-118prim.hepmc3")
endif()
if(CELERITAS_USE_ROOT)
  set(CELERG4_WRITE_SD_HITS "true")
  set(CELERG4_OFFLOAD_FILE "simple-cms.offloaded.root")
else()
  set(CELERG4_WRITE_SD_HITS "false")
  set(CELERG4_OFFLOAD_FILE "simple-cms.offloaded.hepmc3")
endif()
configure_file(
  "simple-cms-test.inp.json.in"
  "simple-cms-test.inp.json" @ONLY
)

set(_env
  ${CELER_G4ENV}
  "CELER_LOG_LOCAL=debug"
  "CELER_DISABLE_PARALLEL=1"
)
if(Geant4_multithreaded_FOUND)
  list(APPEND _env
    "G4FORCENUMBEROFTHREADS=2"
  )
endif()
if(CELERITAS_USE_OpenMP)
  list(APPEND _env "OMP_NUM_THREADS=1")
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_needs_device)
else()
  set(_needs_device DISABLED true)
endif()

set(_props)
if(NOT CELERITAS_USE_HepMC3 OR NOT CELERITAS_USE_Geant4)
  set(_props DISABLED true)
endif()
if(CELERITAS_CORE_GEO STREQUAL "Geant4")
  # FIXME: geant4 navigator doesn't work at scale
  set(_props DISABLED true)
endif()
if(CELERITAS_REAL_TYPE STREQUAL "float")
  # FIXME: propagation fails in single precision
  set(_props DISABLED true)
endif()

### CELERITAS OFFLOADING WITH GPU ###
add_test(NAME "app/celer-g4:gpu"
  COMMAND "$<TARGET_FILE:celer-g4>" "simple-cms-test.inp.json"
)
set_tests_properties("app/celer-g4:gpu" PROPERTIES
  ENVIRONMENT "${_env}"
  RESOURCE_LOCK gpu
  LABELS "app;nomemcheck;gpu"
  ${_props}
  ${_needs_device}
)

### CELERITAS OFFLOADING WITH CPU ###
list(APPEND _env
  "CELER_DISABLE_DEVICE=1"
)
add_test(NAME "app/celer-g4:cpu"
  COMMAND "$<TARGET_FILE:celer-g4>" "simple-cms-test.inp.json"
)
set_tests_properties("app/celer-g4:cpu" PROPERTIES
  ENVIRONMENT "${_env}"
  RESOURCE_LOCK gpu
  LABELS "app;nomemcheck"
  ${_props}
)

### NO CELERITAS OFFLOADING ###
list(APPEND _env
  "CELER_DISABLE=1"
)
add_test(NAME "app/celer-g4:none"
  COMMAND "$<TARGET_FILE:celer-g4>" "simple-cms-test.inp.json"
)
set_tests_properties("app/celer-g4:none" PROPERTIES
  ENVIRONMENT "${_env}"
  LABELS "app;nomemcheck"
  ${_props}
)

#-----------------------------------------------------------------------------#
