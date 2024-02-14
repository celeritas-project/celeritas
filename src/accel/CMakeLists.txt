#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS Celeritas::celeritas Celeritas::corecel)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  AlongStepFactory.cc
  FastSimulationOffload.cc
  GeantSimpleCalo.cc
  GeantStepDiagnostic.cc
  Logger.cc
  LocalTransporter.cc
  SetupOptions.cc
  SetupOptionsMessenger.cc
  SharedParams.cc
  SimpleOffload.cc
  detail/GeantSimpleCaloSD.cc
  detail/HitManager.cc
  detail/HitProcessor.cc
  detail/SensDetInserter.cc
  detail/TouchableUpdater.cc
)

celeritas_polysource(ExceptionConverter)

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  list(APPEND SOURCES TrackingManagerOffload.cc)
endif()

if(CELERITAS_USE_JSON)
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND PRIVATE_DEPS HepMC3::HepMC3)
  list(APPEND SOURCES HepMC3PrimaryGenerator.cc)
endif()

if(CELERITAS_USE_OpenMP)
  list(APPEND PRIVATE_DEPS OpenMP::OpenMP_CXX)
endif()

celeritas_get_g4libs(_g4_private digits_hits run)
list(APPEND PRIVATE_DEPS ${_g4_private})

celeritas_get_g4libs(_g4_public event global track tracking intercoms geometry)
list(APPEND PUBLIC_DEPS ${_g4_public})

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(accel ${SOURCES})
celeritas_target_link_libraries(accel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
