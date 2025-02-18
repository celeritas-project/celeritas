#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_HepMC3)
  set(_needs_hepmc DISABLE)
endif()
if(CELERITAS_CORE_GEO STREQUAL "ORANGE")
  set(_needs_not_orange DISABLE)
endif()

#-----------------------------------------------------------------------------#

celeritas_add_test_library(testcel_accel
  SDTestBase.cc
  SimpleSensitiveDetector.cc
)
celeritas_target_link_libraries(testcel_accel
  PUBLIC
    testcel_celeritas testcel_harness
  PRIVATE
    ${Geant4_LIBRARIES}
)

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES
    testcel_accel testcel_core Celeritas::accel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(ExceptionConverter.test.cc)
celeritas_add_test(HepMC3PrimaryGenerator.test.cc
  ${_needs_hepmc}
  ENVIRONMENT "${CELER_G4ENV}")
celeritas_add_test(RZMapMagneticField.test.cc)
celeritas_add_test(detail/HitManager.test.cc
  ENVIRONMENT "${CELER_G4ENV}")
celeritas_add_test(detail/HitProcessor.test.cc
  ENVIRONMENT "${CELER_G4ENV}")
celeritas_add_test(detail/LevelTouchableUpdater.test.cc
  ${_needs_not_orange})
if(CELERITAS_REAL_TYPE STREQUAL "double")
  # This test requires Geant4 *geometry* which is incompatible
  # with single-precision
  celeritas_add_test(detail/NaviTouchableUpdater.test.cc
    LINK_LIBRARIES testcel_geocel)
endif()

#-----------------------------------------------------------------------------#
