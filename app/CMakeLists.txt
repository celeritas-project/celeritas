#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(BUILD_TESTING)
  if(CELERITAS_USE_Python)
    set(CELER_PYTHON "$<TARGET_FILE:Python::Interpreter>")
    set(CELER_NEEDS_PYTHON)
  else()
    set(CELER_PYTHON "python3")
    set(CELER_NEEDS_PYTHON DISABLED true)
  endif()

  if(CELERITAS_USE_OpenMP)
    set(CELER_OMP_ENV "OMP_NUM_THREADS=4")
    set(CELER_PROCESSORS PROCESSORS 4)
  else()
    set(CELER_OMP_ENV)
    set(CELER_PROCESSORS)
  endif()
  set(CELER_APP_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")

  # Set environment variables from Geant4-exported configuration or
  # environment currently used only for CTest setup
  celeritas_get_g4env(CELER_G4ENV)
  # Set option to disable GPU environment
  celeritas_get_disable_device(CELER_DISABLE_DEVICE)
endif()

#-----------------------------------------------------------------------------#
# Application utils
#-----------------------------------------------------------------------------#

celeritas_add_library(celer_app_utils STATIC
  CliUtils.cc
)

celeritas_target_link_libraries(celer_app_utils
  PUBLIC CLI11::CLI11
  PRIVATE Celeritas::corecel Celeritas::ExtMPI
)

celeritas_target_include_directories(celer_app_utils
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

#-----------------------------------------------------------------------------#
# Geant4/ROOT data interface
#-----------------------------------------------------------------------------#

# Orange JSON updater script
celeritas_add_executable(orange-update orange-update.cc)
celeritas_target_link_libraries(orange-update
  Celeritas::orange
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  celer_app_utils
)

# Exporter
celeritas_add_executable(celer-export-geant celer-export-geant.cc)
celeritas_target_link_libraries(celer-export-geant
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  celer_app_utils
)

add_subdirectory(celer-sim)
add_subdirectory(celer-g4)
add_subdirectory(celer-geo)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()
set(_geant_test_inp "${CELER_APP_DATA_DIR}/four-steel-slabs.gdml")
if(CELERITAS_USE_ROOT AND CELERITAS_USE_Geant4)
  set(_needs_g4_root)
else()
  set(_needs_g4_root DISABLED true)
endif()

add_test(NAME "app/celer-export-geant"
  COMMAND "$<TARGET_FILE:celer-export-geant>"
  "${_geant_test_inp}" "" "test-data.root"
)
set_tests_properties("app/celer-export-geant" PROPERTIES
  ENVIRONMENT "${CELER_G4ENV}"
  REQUIRED_FILES "${_geant_test_inp}"
  LABELS "app"
  ${_needs_g4_root}
)

#-----------------------------------------------------------------------------#
