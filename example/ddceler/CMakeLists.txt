#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#
# DD4hep Comparison Executable
# Note that this does not require Celeritas
#-----------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.18...4.1)
project(ddceler-compare VERSION 0.1.0 LANGUAGES CXX)

find_package(ROOT REQUIRED)
find_package(DD4hep REQUIRED COMPONENTS DDG4)

add_executable(ddceler-compare ddceler-compare.cc)

target_link_libraries(ddceler-compare
  PRIVATE
    ROOT::Core
    ROOT::Hist
    ROOT::Graf
    ROOT::Gpad
    ROOT::Tree
    ROOT::TreePlayer
    ROOT::RIO
    DD4hep::DDCore
    DD4hep::DDG4
)
