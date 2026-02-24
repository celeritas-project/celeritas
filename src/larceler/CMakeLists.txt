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
  ROOT::Tree
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
)

#-----------------------------------------------------------------------------#
# Create mock upstream library for building against official LArSoft
#-----------------------------------------------------------------------------#

set(_larsoft_tool_target larsim::OpticalPropagationTools)
if(NOT TARGET ${_larsoft_tool_target})
  message(WARNING "LArSoft version is too old for integrated Celeritas support: "
    "defining fake plugin target for local testing "
    "(missing target ${_larsoft_tool_target})"
  )
  celeritas_add_interface_library(OpticalPropagationTools)
  celeritas_target_include_directories(OpticalPropagationTools
    SYSTEM INTERFACE
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/larceler/larsim-future>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/larceler/larsim-future>"
  )
  set(_larsoft_tool_target Celeritas::OpticalPropagationTools)
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(larceler
   LarStandaloneRunner.cc
   LarDataReader.cc
)
celeritas_target_link_libraries(larceler
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
# Create plugins
#-----------------------------------------------------------------------------#

macro(celeritas_add_larsoft_plugin target)
  celeritas_add_src_library(${target} MODULE ${target}.cc)
  celeritas_target_link_libraries(${target}
    PRIVATE
      art::Utilities
      cetlib::cetlib
      cetlib_except::cetlib_except
  )
endmacro()

celeritas_add_larsoft_plugin(LarCelerStandalone)
celeritas_target_link_libraries(LarCelerStandalone
  PRIVATE
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
    lardataobj::MCBase
    ${_larsoft_tool_target}
  PUBLIC
    Celeritas::larceler
)

celeritas_add_larsoft_plugin(GeoSimExporterModule)
celeritas_target_link_libraries(GeoSimExporterModule
  PRIVATE
    ROOT::Tree
    art_root_io::TFileService_service
    larcore::ServiceUtil
    larcorealg::Geometry
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
    lardataobj::MCBase
)

#-----------------------------------------------------------------------------#
