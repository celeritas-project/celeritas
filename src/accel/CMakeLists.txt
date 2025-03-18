#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS
  nlohmann_json::nlohmann_json
  Celeritas::ExtOpenMP # TODO: provide thread utils in corecel
)
set(PUBLIC_DEPS
  Celeritas::BuildFlags
  Celeritas::corecel
  Celeritas::geocel
  Celeritas::celeritas
)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  AlongStepFactory.cc
  FastSimulationIntegration.cc
  FastSimulationModel.cc
  GeantSimpleCalo.cc
  GeantStepDiagnostic.cc
  IntegrationBase.cc
  Logger.cc
  LocalTransporter.cc
  CylMapMagneticField.cc
  SetupOptions.cc
  SetupOptionsMessenger.cc
  SharedParams.cc
  SimpleOffload.cc
  UserActionIntegration.cc
  detail/GeantSimpleCaloSD.cc
  detail/IntegrationSingleton.cc
)

celeritas_polysource(ExceptionConverter)

if(Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
  list(APPEND SOURCES
    TrackingManagerIntegration.cc
    TrackingManager.cc
    TrackingManagerConstructor.cc
  )
endif()

if(CELERITAS_USE_HepMC3)
  list(APPEND PRIVATE_DEPS HepMC3::HepMC3)
  list(APPEND SOURCES HepMC3PrimaryGenerator.cc)
endif()

celeritas_get_g4libs(_g4_private digits_hits run)
list(APPEND PRIVATE_DEPS ${_g4_private})

celeritas_get_g4libs(_g4_public event intercoms geometry global track tracking)
list(APPEND PUBLIC_DEPS ${_g4_public})

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(accel ${SOURCES})
celeritas_target_link_libraries(accel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
