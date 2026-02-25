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
# Create library for non-ART components
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
# Create ART modules
#-----------------------------------------------------------------------------#

# Add an `art` plugin: separate the tool definition from the ART instantiation
macro(celeritas_add_art_module basename)
  set(tgt_ "${basename}_module")
  celeritas_add_src_library(${tgt_} MODULE
    "${basename}.cc"
    "${basename}_module.cc"
  )
endmacro()

celeritas_add_art_module(PDFullSimCeler)
celeritas_target_link_libraries(PDFullSimCeler_module
  PUBLIC
    Celeritas::larceler
  PRIVATE
    cetlib::cetlib
    cetlib_except::cetlib_except
    fhiclcpp::fhiclcpp
    art::Utilities
    messagefacility::MF_MessageLogger
    larcore::ServiceUtil
    larcorealg::Geometry
    lardataobj::Simulation
    ${_larsoft_tool_target}
)

celeritas_add_art_module(GeoSimExporter module)
celeritas_target_link_libraries(GeoSimExporter_module
  PRIVATE
    ROOT::Tree
    fhiclcpp::fhiclcpp
    art::Utilities
    art_root_io::TFileService_service
    larcore::ServiceUtil
    larcorealg::Geometry
    lardataobj::Simulation
    Celeritas::corecel
)

#-----------------------------------------------------------------------------#
# Install FHICL module/example files
#-----------------------------------------------------------------------------#

set(_fcl_files
  GeoSimExporter.fcl
  GeoSimExporter_run.fcl
  PDFullSimCeler.fcl
)

# Copy into $BUILD/fcl directory at build time (re-copies when source changes)
set(_fcl_outputs)
foreach(_fcl IN LISTS _fcl_files)
  set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${_fcl}")
  set(_dst "${CELERITAS_FHICL_OUTPUT_DIRECTORY}/${_fcl}")
  add_custom_command(
    OUTPUT "${_dst}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_src}" "${_dst}"
    DEPENDS "${_src}"
    COMMENT "Copying FHICL file: ${_fcl}"
  )
  list(APPEND _fcl_outputs "${_dst}")
endforeach()

add_custom_target(larceler_fcl ALL DEPENDS ${_fcl_outputs})
add_dependencies(larceler larceler_fcl)

# Install to $PREFIX/fcl
install(FILES ${_fcl_files}
  DESTINATION "${CELERITAS_INSTALL_FHICLDIR}"
  COMPONENT runtime
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
