#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_accel
  SDTestBase.cc
  SimpleSensitiveDetector.cc
)
celeritas_target_link_libraries(testcel_accel
  PUBLIC
    Celeritas::testcel_celeritas Celeritas::testcel_harness
  PRIVATE
    ${Geant4_LIBRARIES}
)

celeritas_setup_tests(SERIAL PREFIX accel
  LINK_LIBRARIES
    Celeritas::testcel_accel Celeritas::testcel_core Celeritas::accel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(ExceptionConverter.test.cc)
celeritas_add_test(HepMC3PrimaryGenerator.test.cc
  ENVIRONMENT "${CELERITASTEST_G4ENV}")
celeritas_add_test(RZMapMagneticField.test.cc)
celeritas_add_test(detail/HitManager.test.cc
  ENVIRONMENT "${CELERITASTEST_G4ENV}")
celeritas_add_test(detail/HitProcessor.test.cc
  ENVIRONMENT "${CELERITASTEST_G4ENV}")
if(CELERITAS_REAL_TYPE STREQUAL "double")
  # This test requires Geant4 *geometry* which is incompatible
  # with single-precision
  celeritas_add_test(detail/TouchableUpdater.test.cc
    LINK_LIBRARIES Celeritas::testcel_geocel)
endif()

#-----------------------------------------------------------------------------#
