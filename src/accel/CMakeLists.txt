#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
set(PUBLIC_DEPS Celeritas::celeritas Celeritas::corecel ${Geant4_LIBRARIES})

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  AlongStepFactory.cc
  Logger.cc
  LocalTransporter.cc
  SetupOptions.cc
  SetupOptionsMessenger.cc
  SharedParams.cc
  SimpleOffload.cc
  detail/HitManager.cc
  detail/HitProcessor.cc
)

celeritas_polysource(ExceptionConverter)

if(CELERITAS_USE_JSON)
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND PRIVATE_DEPS HepMC3::HepMC3)
  list(APPEND SOURCES HepMC3PrimaryGenerator.cc)
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(accel ${SOURCES})
celeritas_target_link_libraries(accel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
