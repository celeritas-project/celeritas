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
function(celerias_accel_add_tests target name suite test)
  celeritas_g4_add_tests(${target}
    NAME "${name}:${suite}.${test}"
    ARGS "--gtest_filter=${suite}.${test}"
    LABELS unit
    ${ARGN}
  )
endfunction()

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

  # Add every combination of test for LArSphere.run
  celerias_accel_add_tests("${_target}" "${_name}" LarSphere run)

  if(_basename STREQUAL "TrackingManagerIntegration")
    # Add cpu/disabled for UI
    celerias_accel_add_tests("${_target}" "${_name}" LarSphere ui
      OFFLOAD cpu g4
      RMTYPE serial mt
    )
  endif()

  # Do TestEm3 only with celeritas cpu/gpu and serial run manager
  celerias_accel_add_tests("${_target}" "${_name}" TestEm3 run
    OFFLOAD cpu gpu
    RMTYPE serial
  )
endforeach()

#-----------------------------------------------------------------------------#
