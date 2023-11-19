#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(BUILD_TESTING)
  # Set environment variables from Geant4-exported configuration
  set(CELER_G4ENV)
  foreach(_ds IN LISTS Geant4_DATASETS)
    list(APPEND CELER_G4ENV
      "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
  endforeach()

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

  # Disable device tests if no CUDA/HIP or if environment is set
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    if("$ENV{CELER_DISABLE_DEVICE}")
      message(WARNING
        "Disabling GPU testing because CELER_DISABLE_DEVICE is set"
      )
      set(CELER_DISABLE_DEVICE TRUE)
    else()
      set(CELER_DISABLE_DEVICE FALSE)
    endif()
  else()
    # No device support
    set(CELER_DISABLE_DEVICE TRUE)
  endif()
endif()

#-----------------------------------------------------------------------------#
# Geant4/ROOT data interface
#-----------------------------------------------------------------------------#

# Orange JSON updater script
celeritas_add_executable(orange-update orange-update.cc)
celeritas_target_link_libraries(orange-update
  Celeritas::orange
)
if(CELERITAS_USE_JSON)
  celeritas_target_link_libraries(orange-update
    nlohmann_json::nlohmann_json
  )
endif()

# Exporter
celeritas_add_executable(celer-export-geant celer-export-geant.cc)
celeritas_target_link_libraries(celer-export-geant
  Celeritas::celeritas
)
if(CELERITAS_USE_JSON)
  celeritas_target_link_libraries(celer-export-geant
    nlohmann_json::nlohmann_json
  )
endif()

# Reader
celeritas_add_executable(celer-dump-data celer-dump-data.cc)
celeritas_target_link_libraries(celer-dump-data
  Celeritas::celeritas
)

add_subdirectory(celer-sim)
add_subdirectory(celer-g4)

if(CELERITAS_BUILD_DEMOS)
  add_subdirectory(demo-interactor)
  add_subdirectory(demo-geo-check)
  if(CELERITAS_USE_CUDA)
    add_subdirectory(demo-rasterizer)
  endif()
endif()

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

add_test(NAME "app/celer-dump-data"
  COMMAND "$<TARGET_FILE:celer-dump-data>"
    "test-data.root"
)
set_tests_properties("app/celer-dump-data" PROPERTIES
  DEPENDS "app/celer-export-geant"
  REQUIRED_FILES "test-data.root"
  LABELS "app"
  ${_needs_g4_root}
)

#-----------------------------------------------------------------------------#
