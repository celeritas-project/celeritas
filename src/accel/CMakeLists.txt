#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
  Logger.cc
  LocalTransporter.cc
  SharedParams.cc
)

celeritas_polysource(ExceptionConverter)

if(CELERITAS_USE_JSON)
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
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
