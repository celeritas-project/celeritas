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

# Associate helper targets with test targets for IDEs
set(CMAKE_FOLDER test/accel)

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

set(_optical_args "RMTYPE" "serial")
if(NOT (CELERITAS_CORE_GEO STREQUAL "Geant4"))
  list(APPEND _optical_args "mt")
endif()
if(CELERITAS_CORE_GEO STREQUAL "VecGeom" AND VecGeom_VERSION VERSION_LESS 2.2)
  # VecGeom < 2.2 does not provide surface normals: required for optical
  # (hack the run manager
  list(APPEND _optical_args "DISABLE")
endif()


# When CELERITASTEST_ACCEL_WRAPPER is enabled (see the
# "ndebug-vecgeom-regression" preset in ci-ubuntu-github.json,
# launched by .github/workflows/build-regression.yml), you can wrap the
# googletest call in a script that dumps or compares the output.
# For convenience, you can set the variable to "regression" to compare or "regression-update"
# to force an update.
set(CELERITASTEST_ACCEL_WRAPPER "" CACHE STRING
  "Arguments to add before calls to accel integration test executables (or 'regression' or 'regression-update')"
)
mark_as_advanced(CELERITASTEST_ACCEL_WRAPPER)

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

  # Regression tests require G4 threading to be 1 for reproducibility
  set(CELERITASTEST_G4NT 1)
endif()
if(CELERITASTEST_ACCEL_WRAPPER)
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

function(add_integration_tests basename suite test)
  set(_target "${CELERITASTEST_DIR}_${basename}")
  set(_name "${CELERITASTEST_DIR}/${basename}")

  if(NOT TARGET "${_target}")
    add_executable("${_target}" "${basename}.test.cc")
    add_dependencies(test_accel_integration "${_target}")
    set_target_properties("${_target}" PROPERTIES
      OUTPUT_NAME "${basename}"
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

foreach(_basename IN ITEMS FastSimulationIntegration TrackingManagerIntegration UserActionIntegration)
  # Add every combination of test for LArSphere.run
  add_integration_tests(
    ${_basename} LarSphere run
  )

  # Test OpNovice with optical-friendly run manager types
  add_integration_tests(
    ${_basename} OpNoviceOptical run
    OFFLOAD cpu gpu g4
    ${_optical_args}
  )

  # Do TestEm3 only with celeritas cpu/gpu and serial run manager
  add_integration_tests(
    ${_basename} TestEm3 run
    OFFLOAD cpu gpu
    RMTYPE serial
  )

endforeach()

# Test that Fast Simulation setup fails if particles/regions are not correct
add_integration_tests(
  FastSimulationIntegration LarSphereSetupFailure run
  OFFLOAD cpu
  RMTYPE serial
)

# Test state dependent output
add_integration_tests(
  TrackingManagerIntegration LarSphere state_dep
  OFFLOAD g4
)

# Test that the UI integration works (and ignores if Celeritas is disabled)
add_integration_tests(
  TrackingManagerIntegration LarSphere run_ui
  OFFLOAD cpu g4
  RMTYPE serial mt
)

# Test that the UI integration fails if options aren't set
add_integration_tests(
  TrackingManagerIntegration LarSphere no_set_options
  OFFLOAD cpu
  RMTYPE serial mt
)

# Test automatic state-dependent hooks without manual run action calls
add_integration_tests(
  TrackingManagerIntegration TMIAutoHooks run
  OFFLOAD cpu
  RMTYPE serial mt
)

# Test that legacy manual run action calls are harmless and warn
add_integration_tests(
  TrackingManagerIntegration LegacyTMIHooks run
  OFFLOAD cpu
  RMTYPE serial mt
)

# Test optical
add_integration_tests(
  TrackingManagerIntegration LarSphereOptical run
  OFFLOAD cpu gpu g4
  ${_optical_args}
)

# Test with optical physics offloading distributions
add_integration_tests(
  UserActionIntegration LarSphereOpticalTrackOffload run
  OFFLOAD cpu gpu g4
  ${_optical_args}
)

# Test optical surface physics
add_integration_tests(
  TrackingManagerIntegration OpticalSurfaces run
  OFFLOAD cpu gpu g4
  ${_optical_args}
)

# Test cleanup after runtime error
add_integration_tests(
  TrackingManagerIntegration TestEm3 max_step_iters_error
  OFFLOAD cpu gpu
  RMTYPE serial
)

# Test that we can safely flush and restore primaries multiple times per event
add_integration_tests(
  TrackingManagerIntegration WaterSphere run_small_flush
  OFFLOAD cpu gpu
  RMTYPE serial mt
)

# Test that primary batches can exceed the initial initializer capacity
add_integration_tests(
  TrackingManagerIntegration WaterSphere run_small_initializers
  OFFLOAD cpu gpu
  RMTYPE serial mt
)

# Test with optical physics offloading distributions
add_integration_tests(
  UserActionIntegration LarSphereOpticalOffload run
  OFFLOAD cpu gpu g4
  ${_optical_args}
)

#-----------------------------------------------------------------------------#
