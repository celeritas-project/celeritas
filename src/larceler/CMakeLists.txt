#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  fhiclcpp::fhiclcpp
  Celeritas::geocel
  Celeritas::celeritas
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
  larcoreobj::SimpleTypesAndConstants
  lardataobj::MCBase
)

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(larceler
   inp/LarStandaloneRunner.cc
   LarStandaloneRunner.cc
)
celeritas_target_link_libraries(larceler
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
# Create plugin
#-----------------------------------------------------------------------------#

set(_plugin_name LarCelerStandalone)

celeritas_add_src_library(${_plugin_name} MODULE ${_plugin_name}.cc)
celeritas_target_link_libraries(${_plugin_name}
  PRIVATE
    art::Utilities
    cetlib::cetlib
    cetlib_except::cetlib_except
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
