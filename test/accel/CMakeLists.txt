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

function(celeritas_add_integration_tests basename suite test)
  set(_target "${CELERITASTEST_DIR}_${basename}")
  set(_name "${CELERITASTEST_DIR}/${basename}")

  if(NOT TARGET "${_target}")
    add_executable(${_target} "${basename}.test.cc")
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
    ARGS "--gtest_filter=${suite}.${test}"
    LABELS unit
    ${ARGN}
    ${fail_if_unsupported}
  )
endfunction()

#-----------------------------------------------------------------------------#

foreach(_basename IN ITEMS UserActionIntegration TrackingManagerIntegration)
  # Add every combination of test for LArSphere.run
  celeritas_add_integration_tests(${_basename} LarSphere run)

  # Do TestEm3 only with celeritas cpu/gpu and serial run manager
  celeritas_add_integration_tests(${_basename} TestEm3 run
    OFFLOAD cpu gpu
    RMTYPE serial
  )
  # Test OpNovice
  celeritas_add_integration_tests(
    ${_basename} OpNoviceOptical run
    OFFLOAD cpu gpu g4
    RMTYPE ${_optical_rm_type})
endforeach()

# Test that the UI integration works (and ignores if Celeritas is disabled)
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphere run_ui
  OFFLOAD cpu g4
  RMTYPE serial mt
)

set(_optical_rm_type "serial")
if(NOT (CELERITAS_CORE_GEO STREQUAL "Geant4"))
  list(APPEND _optical_rm_type "mt")
endif()

# Test with optical physics, except that currently trying to use Geant4
# navigation for the optical tracks wreaks havoc
celeritas_add_integration_tests(
  TrackingManagerIntegration LarSphereOptical run
  OFFLOAD cpu gpu g4
  RMTYPE ${_optical_rm_type}
)

#-----------------------------------------------------------------------------#
