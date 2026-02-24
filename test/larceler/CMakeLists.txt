#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES
    Celeritas::larceler
    fhiclcpp::fhiclcpp
    lardataobj::MCBase
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(LarStandaloneRunner.test.cc)
celeritas_add_test(LarDataReader.test.cc LINK_LIBRARIES Celeritas::celeritas)

#-----------------------------------------------------------------------------#
