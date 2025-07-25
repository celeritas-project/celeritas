#------------------------------- -- cmake -- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.18)
project(CeleritasDD4hepExample VERSION 0.0.1 LANGUAGES CXX)
cmake_policy(VERSION 3.12...3.30)

# Find DD4hep
find_package(DD4hep REQUIRED COMPONENTS DDG4 DDRec DDAlign)

# Include DD4hep
include(${DD4hep_DIR}/cmake/DD4hep.cmake)

# Add the detector plugin
dd4hep_add_plugin(CeleritasDD4hep
  SOURCES celeritas-dd4hep.cc
  USES DD4hep::DDCore DD4hep::DDRec
)

# Install the plugin
install(TARGETS CeleritasDD4hep
  LIBRARY DESTINATION lib
)

# Install the geometry files
install(FILES
  celeritas-dd4hep.xml
  DESTINATION share/CeleritasDD4hep
)
