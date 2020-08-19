#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_ROOT)
  include(${ROOT_USE_FILE})

  add_executable(geant-exporter geant-exporter/geant-exporter.cc
    geant-exporter/ActionInitialization.cc
    geant-exporter/DetectorConstruction.cc
    geant-exporter/PrimaryGeneratorAction.cc)

  target_link_libraries(geant-exporter
    Celeritas::Core
    Celeritas::IO
    ROOT::Core
    ROOT::Tree
    XercesC::XercesC
    ${Geant4_LIBRARIES})

endif()

#-----------------------------------------------------------------------------#
