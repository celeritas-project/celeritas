#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  fhiclcpp::fhiclcpp
  larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
  lardataobj::MCBase
  Celeritas::geocel
  Celeritas::celeritas
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
)

if(TARGET larsim::OpticalPropagationTools)
  message(SEND_ERROR "This version of Celeritas is meant for testing larsim "
    "before https://github.com/nuRiceLab/larsim/pull/1 : please implement or "
    "update Celeritas")
endif()

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
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
    lardataobj::MCBase
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
