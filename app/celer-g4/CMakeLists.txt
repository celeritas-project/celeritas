#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4)
  set(SOURCES
    celer-g4.cc
    ActionInitialization.cc
    DetectorConstruction.cc
    EventAction.cc
    GeantDiagnostics.cc
    GlobalSetup.cc
    PrimaryGeneratorAction.cc
    RunAction.cc
    SensitiveDetector.cc
    SensitiveHit.cc
    TrackingAction.cc
  )
  set(LIBRARIES Celeritas::accel)
else()
  set(SOURCES
    celer-g4.nogeant.cc
  )
  set(LIBRARIES)
endif()

if(CELERITAS_USE_ROOT AND CELERITAS_USE_Geant4)
  function(celeritas_local_root_generate_dictionary name)
    # No need for the rootmap file for an executable.
    set(CMAKE_ROOTTEST_NOROOTMAP TRUE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CELERITAS_RUNTIME_OUTPUT_DIRECTORY})
    include_directories(
      # root_generate_dictionary cannot handle the '$<INSTALL_INTERFACE:include>'
      # part of the Celeritas::corecel target's include, so we cannot use it in
      # the 'DEPENDENCIES' property.
      "${CELERITAS_HEADER_CONFIG_DIRECTORY}"
      "${PROJECT_SOURCE_DIR}/src"
      # The include directory are attached to the Geant4 target through
      # INTERFACE_INCLUDE_DIRECTORIES which is not looked at by
      # root_generate_dictionary.
      ${Geant4_INCLUDE_DIRS}
      ${CLHEP_INCLUDE_DIRS}
    )
    root_generate_dictionary(${name}
      ${CMAKE_CURRENT_SOURCE_DIR}/HitRootIO.hh
      ${CMAKE_CURRENT_SOURCE_DIR}/SensitiveHit.hh
      NOINSTALL
      MODULE celer-g4
      # TODO Uncomment this line and remove the include_directories when
      # the minimal required ROOT version is v6.28/02 or later.
      # DEPENDENCIES ${Geant4_LIBRARIES} Celeritas::accel Celeritas::corecel
      LINKDEF "${CMAKE_CURRENT_SOURCE_DIR}/HitClassesLinkDef.h"
    )
  endfunction(celeritas_local_root_generate_dictionary)
  celeritas_local_root_generate_dictionary(HitClassesRootInterfaces)
  list(APPEND SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/HitClassesRootInterfaces.cxx"
    HitRootIO.cc
  )
  list(APPEND LIBRARIES ROOT::Tree)
endif()

celeritas_add_executable(celer-g4 ${SOURCES})
celeritas_target_link_libraries(celer-g4
  ${LIBRARIES}
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_BUILD_TESTS)
  return()
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CELERG4_MAX_NUM_TRACKS 524288)
    set(CELERG4_INIT_CAPACITY 4194304)
  else()
    # Use smaller number of tracks when running on CPU
    set(CELERG4_MAX_NUM_TRACKS 2048)
    set(CELERG4_INIT_CAPACITY 65536)
  endif()
else()
  # Use smaller number of tracks when running on CPU
  set(CELERG4_MAX_NUM_TRACKS 1024)
  set(CELERG4_INIT_CAPACITY 32768)
endif()

if(NOT CELERITAS_DEBUG)
  set(CELERG4_EVENTS
    "${CELER_APP_DATA_DIR}/ttbarsplit-12evt-1108prim.hepmc3")
else()
  # Use fewer events for debug
  set(CELERG4_EVENTS
    "${CELER_APP_DATA_DIR}/ttbarbits-12evt-118prim.hepmc3")
endif()
if(CELERITAS_USE_ROOT)
  set(CELERG4_ROOT_OPTIONS "/celerg4/rootBufferSize 128000
/celerg4/writeSDHits true")
endif()
configure_file(
  "simple-cms-test.mac.in"
  "simple-cms-test.mac" @ONLY
)
add_test(NAME "app/celer-g4"
  COMMAND "$<TARGET_FILE:celer-g4>"
  "simple-cms-test.mac"
)
set(_env
  ${CELER_G4ENV}
  "CELER_LOG_LOCAL=debug"
  "CELER_DISABLE_PARALLEL=1"
)
if(Geant4_multithreaded_FOUND)
  list(APPEND _env
    "G4FORCENUMBEROFTHREADS=2"
  )
endif()
if(CELERITAS_USE_OpenMP)
  list(APPEND _env "OMP_NUM_THREADS=1")
endif()

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  set(_props RESOURCE_LOCK gpu)
  set(_labels app gpu)
else()
  set(_props)
  set(_labels app)
endif()
if(NOT CELERITAS_USE_HepMC3 OR NOT CELERITAS_USE_Geant4)
  list(APPEND _props DISABLED true)
endif()

set_tests_properties("app/celer-g4" PROPERTIES
  ENVIRONMENT "${_env}"
  LABELS "${_labels}"
  ${_props}
)

#-----------------------------------------------------------------------------#
