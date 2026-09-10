#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()

if(NOT CELERITAS_USE_covfie)
  set(_needs_covfie DISABLE)
endif()

include(CeleritasG4Tests)

#-----------------------------------------------------------------------------#
# Library
celeritas_get_g4libs(_g4_libs
  global track tracking run event intercoms digits_hits geometry tasking physicslists
)
list(APPEND PUBLIC_DEPS ${_g4_libs})

celeritas_add_test_library(testcel_accel
  DistOffloadMixin.cc
  IntegrationTestBase.cc
)
celeritas_target_link_libraries(testcel_accel
  PUBLIC
    testcel_celeritas_g4 testcel_harness Celeritas::accel
  PRIVATE
    ${_g4_libs}
)

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES
    testcel_accel testcel_celeritas testcel_core Celeritas::accel
)

#-----------------------------------------------------------------------------#
# Variables

set(_optical_rm_type "serial")
if(NOT (CELERITAS_CORE_GEO STREQUAL "Geant4"))
  list(APPEND _optical_rm_type "mt")
endif()


# When CELERITASTEST_ACCEL_WRAPPER is enabled (see the
# "ndebug-vecgeom-regression" preset in ci-ubuntu-github.json,
# launched by .github/workflows/build-regression.yml), you can wrap the
# googletest call in a script that dumps or compares the output.
# For convenience, you can set the variable to "regression" to compare or "regression-update"
# to force an update.
set(CELERITASTEST_ACCEL_WRAPPER "" CACHE STRING
  "Arguments to add before calls to accel integration test executables"
)
if(CELERITASTEST_ACCEL_WRAPPER MATCHES "^regression(-update)?$")
  set(_force)
  if(CMAKE_MATCH_1)
    set(_force "--force")
  endif()
  set(CELERITASTEST_ACCEL_WRAPPER
    "${PROJECT_SOURCE_DIR}/scripts/ci/run-output-regression.py"
    "--source-dir" "${PROJECT_SOURCE_DIR}" # Relative to Celeritas
    "--build-dir" "${CMAKE_BINARY_DIR}" # In top-level build directory
    ${_force}
    "--"
  )
endif()
if(CELERITASTEST_ACCEL_WRAPPER)
  # Note that for regression tests, you should always set G4 threading to 1:
  # otherwise results will not reproduce
  message(VERBOSE "Using wrapper script for accel integration tests "
    "with CELERITASTEST_G4NT=${CELERITASTEST_G4NT}: "
    "${CELERITASTEST_ACCEL_WRAPPER}"
  )
endif()

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(ExceptionConverter.test.cc)
celeritas_add_test(HepMC3PrimaryGenerator.test.cc
  ${_needs_hepmc}
  ENVIRONMENT "${CELER_G4ENV}")

# Fields
celeritas_add_test(CartMapMagneticField.test.cc ${_needs_covfie})
celeritas_add_test(CylMapMagneticField.test.cc)

#-----------------------------------------------------------------------------#
# INTEGRATION TESTS
#-----------------------------------------------------------------------------#

add_custom_target(test_accel_integration
  COMMENT "Build accel integration test executables"
)

function(celeritas_add_integration_tests basename suite test)
  set(_target "${CELERITASTEST_DIR}_${basename}")
  set(_name "${CELERITASTEST_DIR}/${basename}")

  if(NOT TARGET "${_target}")
    add_executable("${_target}" "${basename}.test.cc")
    add_dependencies(test_accel_integration "${_target}")
    set_property(TARGET ${_target}
      PROPERTY OUTPUT_NAME "${basename}"
    )
    celeritas_target_link_libraries(${_target}
      ${CELERITASTEST_LINK_LIBRARIES}
      ${_g4_libs}
      testcel_harness
    )
  endif()

  if(Geant4_VERSION VERSION_LESS 11.0 AND basename MATCHES "TrackingManager")
    set(fail_if_unsupported WILL_FAIL)
  elseif(Geant4_VERSION VERSION_LESS 11.1 AND basename MATCHES "FastSim")
    set(fail_if_unsupported WILL_FAIL)
  else()
    set(fail_if_unsupported)
  endif()

  celeritas_g4_add_tests(${_target}
    NAME "${_name}:${suite}.${test}"
    WRAP "${CELERITASTEST_ACCEL_WRAPPER}"
    ARGS "--gtest_filter=${suite}.${test}"
    LABELS unit integration
    ${ARGN}
    ${fail_if_unsupported}
  )
endfunction()

#-----------------------------------------------------------------------------#

foreach(_basename IN ITEMS UserActionIntegration TrackingManagerIntegration FastSimulationIntegration)
  # Add every combination of test for LArSphere.run
  celeritas_add_integration_tests(
    ${_basename} LarSphere run
  )

  # Do TestEm3 only with celeritas cpu/gpu and serial run manager
  celeritas_add_integration_tests(
    ${_basename} TestEm3 run
    OFFLOAD cpu gpu
    RMTYPE serial
  )

  # Test OpNovice with optical-friendly run manager types
  celeritas_add_integration_tests(
    ${_basename} OpNoviceOptical run
    OFFLOAD cpu gpu g4
    RMTYPE ${_optical_rm_type}
  )
endforeach()

# Test state dependent output
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphere state_dep
  OFFLOAD g4
)

# Test that the UI integration works (and ignores if Celeritas is disabled)
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphere run_ui
  OFFLOAD cpu g4
  RMTYPE serial mt
)

# Test that the UI integration fails if options aren't set
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphere no_set_options
  OFFLOAD cpu
  RMTYPE serial mt
)

# Test optical
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphereOptical run
  OFFLOAD cpu gpu g4
  RMTYPE ${_optical_rm_type}
)

# Test with optical physics offloading distributions
celeritas_add_integration_tests(
  UserActionIntegration LarSphereOpticalOffload run
  OFFLOAD cpu gpu g4
  RMTYPE ${_optical_rm_type}
)

# Test optical surface physics
celeritas_add_integration_tests(
  TrackingManagerIntegration OpticalSurfaces run
  OFFLOAD cpu gpu g4
  RMTYPE ${_optical_rm_type}
)

# Test with optical physics offloading distributions
celeritas_add_integration_tests(
  UserActionIntegration LarSphereOpticalTrackOffload run
  OFFLOAD cpu gpu g4
  RMTYPE ${_optical_rm_type}
)

# Test that we can safely flush and restore primaries multiple times per event
celeritas_add_integration_tests(
  TrackingManagerIntegration WaterSphere run_small_flush
  OFFLOAD cpu gpu
  RMTYPE serial mt
)

# Test primary admission with a larger secondary stack
celeritas_add_integration_tests(
  TrackingManagerIntegration TestEm3Rayleigh run_small_capacity
  OFFLOAD cpu gpu
  RMTYPE serial
)

#-----------------------------------------------------------------------------#
