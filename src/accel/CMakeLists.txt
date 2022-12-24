#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::DeviceToolkit)
set(PUBLIC_DEPS Celeritas::celeritas Celeritas::corecel ${Geant4_LIBRARIES})

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  ExceptionConverter.cc
  Logger.cc
  LocalTransporter.cc
  SharedParams.cc
  detail/HitManager.cc
  detail/HitProcessor.cc
)

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(accel ${SOURCES})
celeritas_target_link_libraries(accel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
