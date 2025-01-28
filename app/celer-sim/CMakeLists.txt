#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  celer-sim.cc
  Runner.cc
  RunnerOutput.cc
  RunnerInput.cc
  RunnerInputIO.json.cc
  Transporter.cc
)
set(LIBRARIES
  Celeritas::celeritas
  Celeritas::DeviceToolkit
  nlohmann_json::nlohmann_json
)

if(CELERITAS_USE_CUDA AND CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND LIBRARIES VecGeom::vecgeom)
endif()
if(CELERITAS_USE_OpenMP)
  list(APPEND LIBRARIES OpenMP::OpenMP_CXX)
endif()
if(CELERITAS_USE_ROOT)
  list(APPEND SOURCES RootOutput.cc)
  list(APPEND LIBRARIES ROOT::Core ROOT::Tree)
endif()

# Add the executable
celeritas_add_executable(celer-sim ${SOURCES})
celeritas_target_link_libraries(celer-sim ${LIBRARIES})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

#-----------------------------------------------------------------------------#
function(celer_sim_test test_ext gdml_inp_name hepmc3_inp_name mctruth_name)
  set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/simple-driver.py")
  set(_gdml_inp "${CELER_APP_DATA_DIR}/${gdml_inp_name}")
  set(_hepmc3_inp "${CELER_APP_DATA_DIR}/${hepmc3_inp_name}")
  set(_mctruth_out "")
  set(_geant_exporter_env "CELER_EXPORT_GEANT_EXE=")
  if(CELERITAS_USE_ROOT)
    set(_geant_exporter_env
      "CELER_EXPORT_GEANT_EXE=$<TARGET_FILE:celer-export-geant>")
  endif()
  set(_env
    "CELERITAS_EXE=$<TARGET_FILE:celer-sim>"
    "${_geant_exporter_env}"
    "CELER_DISABLE_PARALLEL=1"
    "CELER_CORE_GEO=${CELERITAS_CORE_GEO}"
  )

  if(CELERITAS_USE_Geant4 AND CELERITAS_USE_HepMC3 AND CELERITAS_USE_Python
      AND (NOT CELERITAS_CORE_GEO STREQUAL "Geant4")
      AND (VecGeom_GDML_FOUND OR NOT (CELERITAS_CORE_GEO STREQUAL "VecGeom")))
    set(_needs_deps)
  else()
    set(_needs_deps DISABLED true)
  endif()
  if(CELER_DISABLE_DEVICE)
    set(_needs_device DISABLED true)
  else()
    set(_needs_device)
  endif()

  if(CELERITAS_REAL_TYPE STREQUAL "float")
    # FIXME: propagation fails in single precision
    set(_needs_deps DISABLED true)
  endif()

  set(_name "app/celer-sim:${test_ext}:gpu")
  add_test(NAME "${_name}"
    COMMAND "${CELER_PYTHON}"
    "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
  )
  set_tests_properties("${_name}" PROPERTIES
    ENVIRONMENT "${_env};${CELER_G4ENV}"
    RESOURCE_LOCK gpu
    REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
    LABELS "app;nomemcheck;gpu"
    ${_needs_deps}
    ${_needs_device}
    ${CELER_NEEDS_PYTHON}
  )

  # Enable ROOT output for the celer-sim-cpu test when ROOT is available
  if(CELERITAS_USE_ROOT)
    set(_mctruth_out "${CMAKE_CURRENT_BINARY_DIR}/${mctruth_name}")
  endif()
  set(CELER_GEO ${CELERITAS_CORE_GEO})
  if(CELERITAS_CORE_GEO STREQUAL "ORANGE" AND NOT CELERITAS_USE_Geant4)
    # Signal to the app that we have to use the JSON input
    set(CELER_GEO "${CELER_CORE_GEO}-JSON")
  endif()

  set(_name "app/celer-sim:${test_ext}:cpu")
  add_test(NAME "${_name}"
    COMMAND "${CELER_PYTHON}"
    "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
  )
  set(_env
    "CELERITAS_EXE=$<TARGET_FILE:celer-sim>"
    "${_geant_exporter_env}"
    "CELER_LOG=debug"
    "CELER_LOG_LOCAL=debug"
    "CELER_DISABLE_DEVICE=1"
    "CELER_DISABLE_PARALLEL=1"
    "CELER_CORE_GEO=${CELERITAS_CORE_GEO}"
  )
  if(CELERITAS_USE_OpenMP AND CELERITAS_USE_ROOT)
    # TODO: ROOT writer is not currently thread safe
    set(CELER_OMP_ENV "OMP_NUM_THREADS=1")
    set(CELER_PROCESSORS)
  endif()
  set_tests_properties("${_name}" PROPERTIES
    ENVIRONMENT "${_env};${CELER_G4ENV};${CELER_OMP_ENV}"
    REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
    LABELS "app;nomemcheck"
    ${CELER_PROCESSORS}
    ${_needs_deps}
  )
endfunction()

#-----------------------------------------------------------------------------#

celer_sim_test(cms
  simple-cms.gdml gamma-4evt-15prim.hepmc3 simple-cms-mctruth.root
)

# TODO: Currently simple-driver.py uses the same Runner input parameters on
#       both celer-sim-cms and celer-sim-lar tests. Some may need to be updated.
celer_sim_test(lar
  lar-sphere.gdml gamma-4evt-15prim.hepmc3 lar-mctruth.root
)

#-----------------------------------------------------------------------------#
