#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()

#-----------------------------------------------------------------------------#

celeritas_add_test_library(testcel_accel
  IntegrationTestBase.cc
)
celeritas_target_link_libraries(testcel_accel
  PUBLIC
    testcel_celeritas testcel_harness
  PRIVATE
    ${Geant4_LIBRARIES}
)

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES
    testcel_core testcel_celeritas Celeritas::accel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(ExceptionConverter.test.cc)
celeritas_add_test(HepMC3PrimaryGenerator.test.cc
  ${_needs_hepmc}
  ENVIRONMENT "${CELER_G4ENV}")
celeritas_add_test(RZMapMagneticField.test.cc)

# High-level integration tests
celeritas_add_test(UserActionIntegration.test.cc
  LINK_LIBRARIES ${Geant4_LIBRARIES})
celeritas_add_test(TrackingManagerIntegration.test.cc
  LINK_LIBRARIES ${Geant4_LIBRARIES})


#-----------------------------------------------------------------------------#
