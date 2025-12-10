#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# See cetmodules/*/Modules
# /cvmfs/larsoft.opensciencegrid.org/products/cetmodules/v3_24_01/Modules
include(CetCMakeEnv)
include(BasicPlugin)

cet_cmake_env()

#-----------------------------------------------------------------------------#
# TODO: this will need to be `lar::OpticalSimInterface`
set(_tool_interface art::tool)

cet_build_plugin(LarCelerStandalone ${_tool_interface}
  LIBRARIES PRIVATE
  Celeritas::geocel
  Celeritas::BuildFlags
  Celeritas::celeritas
)

#-----------------------------------------------------------------------------#
