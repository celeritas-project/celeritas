#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

# NOTE: the library should *not* contain any `art` dependencies
set(SOURCES)
set(PRIVATE_DEPS
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
  celeritas_target_link_libraries(OpticalPropagationTools
    INTERFACE
      CLHEP::Random
      lardataobj::Simulation
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

# Add an `art` plugin module:
# - Target name *must* end with _tool or _module
# - A .fcl file with the same name
macro(celeritas_add_art_plugin target)
  celeritas_add_src_library(${target} MODULE ${target}.cc)

  # Copy into $BUILD/fcl directory
  file(COPY "${target}.fcl"
    DESTINATION "${CELERITAS_FHICL_OUTPUT_DIRECTORY}"
  )
  # Install to $PREFIX/fcl
  install(FILES "${target}.fcl"
    DESTINATION "${CELERITAS_INSTALL_FHICLDIR}"
    COMPONENT runtime
  )
endmacro()

celeritas_add_art_plugin(LarCelerStandalone_tool)
celeritas_target_link_libraries(LarCelerStandalone_tool
  PRIVATE
    cetlib::cetlib
    cetlib_except::cetlib_except
    fhiclcpp::fhiclcpp
    art::Utilities
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
    lardataobj::MCBase
    ${_larsoft_tool_target}
  PUBLIC
    Celeritas::larceler
)

celeritas_add_art_plugin(GeoSimExporter_module)
celeritas_target_link_libraries(GeoSimExporter_module
  PRIVATE
    ROOT::Tree
    fhiclcpp::fhiclcpp
    art::Utilities
    art_root_io::TFileService_service
    larcore::ServiceUtil
    larcorealg::Geometry
    larcoreobj::SimpleTypesAndConstants # Implicit dependency needed by MCBase
    lardataobj::MCBase
)

#-----------------------------------------------------------------------------#
# Set up and install environment script
#-----------------------------------------------------------------------------#

if(APPLE)
  set(CELER_LDVAR DYLD_LIBRARY_PATH)
else()
  set(CELER_LDVAR LD_LIBRARY_PATH)
endif()

set(_filename "larceler-env")
# Configure to run from build directory
set(CELER_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/${_filename}.in"
  "${CELER_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/larceler-env"
  @ONLY
)

# Set up script to run from dynamic installation prefix after install
install(CODE "
  set(CELER_INSTALL_PREFIX \"\$(cd \\\"\$(dirname \\\"$0\\\")\\\"/.. && pwd)\")
  set(CELER_LDVAR \"${CELER_LDVAR}\")
  set(CMAKE_INSTALL_LIBDIR \"${CMAKE_INSTALL_LIBDIR}\")
  set(CELERITAS_INSTALL_FHICLDIR \"${CELERITAS_INSTALL_FHICLDIR}\")
  message(STATUS \"Configuring: \${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${_filename}\")
  configure_file(
    \"${CMAKE_CURRENT_SOURCE_DIR}/${_filename}.in\"
    \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/${_filename}\"
    @ONLY
  )
")

#-----------------------------------------------------------------------------#
