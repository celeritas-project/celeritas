#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()

include(CeleritasG4Tests)

#-----------------------------------------------------------------------------#
celeritas_get_g4libs(_g4_libs
  global track tracking run event intercoms digits_hits geometry tasking
)
list(APPEND PUBLIC_DEPS ${_g4_libs})

celeritas_add_test_library(testcel_accel
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
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(ExceptionConverter.test.cc)
celeritas_add_test(HepMC3PrimaryGenerator.test.cc
  ${_needs_hepmc}
  ENVIRONMENT "${CELER_G4ENV}")
celeritas_add_test(RZMapMagneticField.test.cc)

#-----------------------------------------------------------------------------#
# INTEGRATION TESTS
#-----------------------------------------------------------------------------#
set(_integration_tests
  UserActionIntegration
  TrackingManagerIntegration
)

# FIXME: test harness activate_device clashes with accel
# (use a private CeleritasG4Tests variable to disable till the next PR)
set(_disable_device TRUE)

foreach(_basename IN LISTS _integration_tests)
  # Create the executable
  set(_target "${CELERITASTEST_DIR}_${_basename}")
  set(_name "${CELERITASTEST_DIR}/${_basename}")

  add_executable(${_target} "${_basename}.test.cc")
  set_property(TARGET ${_target}
    PROPERTY OUTPUT_NAME "${_basename}"
  )
  celeritas_target_link_libraries(${_target}
    ${CELERITASTEST_LINK_LIBRARIES}
    ${_g4_libs}
    testcel_harness
  )

  # Add every combination of test for the LAr sphere
  foreach(_suite LarSphere)
    foreach(_test run)
      celeritas_g4_add_tests(${_target}
        NAME "${_name}:${_suite}.${_test}"
        ARGS "--gtest_filter=${_suite}.${_test}"
        LABELS unit
      )
    endforeach()
  endforeach()

  # Do TestEm3 only with celeritas cpu serial
  celeritas_g4_add_one_test(
    "${_name}:TestEm3.run:cpu:serial"
    ${_target}
    "--gtest_filter=TestEm3.run"
    ""
    cpu serial
  )
endforeach()

#-----------------------------------------------------------------------------#
