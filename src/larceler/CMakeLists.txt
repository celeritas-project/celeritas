#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  fhiclcpp::fhiclcpp
  art::Utilities
  cetlib::cetlib
  cetlib_except::cetlib_except
  larcoreobj::SimpleTypesAndConstants
  lardataobj::MCBase
  Celeritas::geocel
  Celeritas::celeritas
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
)

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#
set(_plugin_name LarCelerStandalone)

celeritas_add_src_library(${_plugin_name} MODULE ${_plugin_name}.cc)
celeritas_target_link_libraries(${_plugin_name}
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
