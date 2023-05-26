#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES
  demo-loop.cc
  RunnerInputIO.json.cc
  Runner.cc
  RunnerOutput.cc
  Transporter.cc
)
set(LIBRARIES
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
  Celeritas::DeviceToolkit
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
add_executable(demo-loop ${SOURCES})
celeritas_target_link_libraries(demo-loop ${LIBRARIES})

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_BUILD_TESTS)
  return()
endif()

set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/simple-driver.py")
set(_gdml_inp "${CELER_APP_DATA_DIR}/simple-cms.gdml")
set(_hepmc3_inp "${CELER_APP_DATA_DIR}/gamma-3evt-15prim.hepmc3")
set(_mctruth_out "")
set(_geant_exporter_env "CELER_EXPORT_GEANT_EXE=")
if(CELERITAS_USE_ROOT)
  set(_geant_exporter_env
    "CELER_EXPORT_GEANT_EXE=$<TARGET_FILE:celer-export-geant>")
endif()
add_test(NAME "app/demo-loop:device"
  COMMAND "${CELER_PYTHON}"
  "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
)
set(_env
  "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
  "${_geant_exporter_env}"
  "CELER_DISABLE_PARALLEL=1"
)
if(NOT CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND _env "CELER_DISABLE_VECGEOM=1")
endif()

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_HepMC3 AND CELERITAS_USE_Python)
  set(_needs_deps)
else()
  set(_needs_deps DISABLED true)
endif()
if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_needs_device)
else()
  set(_needs_device DISABLED true)
endif()

set_tests_properties("app/demo-loop:device" PROPERTIES
  ENVIRONMENT "${_env};${CELER_G4ENV}"
  RESOURCE_LOCK gpu
  REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
  LABELS "app;nomemcheck;gpu"
  ${_needs_deps}
  ${_needs_device}
)

# Enable ROOT output for the demo-loop-cpu test when ROOT is available
if(CELERITAS_USE_ROOT)
  set(_mctruth_out "${CMAKE_CURRENT_BINARY_DIR}/mctruth_output.root")
endif()
add_test(NAME "app/demo-loop:host"
  COMMAND "${CELER_PYTHON}"
  "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
)
set(_env
  "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
  "${_geant_exporter_env}"
  "CELER_LOG=debug"
  "CELER_DISABLE_DEVICE=1"
  "CELER_DISABLE_PARALLEL=1"
)
if(CELERITAS_USE_OpenMP)
  # TODO: update when OpenMP nested parallelism is enabled
  if(CELERITAS_USE_ROOT)
    # TODO: ROOT writer is not currently thread safe
    list(APPEND _env "OMP_NUM_THREADS=1")
  else()
    list(APPEND _env "OMP_NUM_THREADS=4")
  endif()
endif()
if(NOT CELERITAS_CORE_GEO STREQUAL "VecGeom")
  list(APPEND _env "CELER_DISABLE_VECGEOM=1")
endif()
set_tests_properties("app/demo-loop:host" PROPERTIES
  ENVIRONMENT "${_env};${CELER_G4ENV}"
  REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
  LABELS "app;nomemcheck"
  ${CELER_PROCESSORS}
  ${_needs_deps}
)

#-----------------------------------------------------------------------------#
