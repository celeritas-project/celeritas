#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND NOT CELERITAS_USE_JSON)
  message(SEND_ERROR "JSON support is required for demos: setting "
    "CELERITAS_BUILD_DEMOS=OFF for next build")
  set(CELERITAS_BUILD_DEMOS OFF CACHE BOOL
    "Disabled because of missing JSON" FORCE)
endif()

if(CELERITAS_BUILD_TESTS)
  # Set environment variables from Geant4-exported configuration
  set(_geant_test_env)
  foreach(_ds IN LISTS Geant4_DATASETS)
    list(APPEND _geant_test_env
      "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
  endforeach()

  if(CELERITAS_USE_Python)
    set(_python_exe "$<TARGET_FILE:Python::Interpreter>")
    set(_disabled_unless_python)
  else()
    set(_python_exe "python3")
    set(_disabled_unless_python DISABLED true)
  endif()

  set(_omp_env)
  set(_processors)
  if(CELERITAS_USE_OpenMP)
    set(_omp_env "OMP_NUM_THREADS=4")
    set(_processors PROCESSORS 4)
  endif()
endif()

#-----------------------------------------------------------------------------#
# Geant4/ROOT data interface
#-----------------------------------------------------------------------------#

# Exporter
add_executable(celer-export-geant celer-export-geant.cc)
celeritas_target_link_libraries(celer-export-geant
  Celeritas::celeritas
)
if(CELERITAS_USE_JSON)
  celeritas_target_link_libraries(celer-export-geant
    nlohmann_json::nlohmann_json
  )
endif()

# Reader
add_executable(celer-dump-data celer-dump-data.cc)
celeritas_target_link_libraries(celer-dump-data
  Celeritas::celeritas
)

if(CELERITAS_USE_ROOT AND CELERITAS_USE_Geant4 AND CELERITAS_BUILD_TESTS)
  set(_geant_test_inp "${CMAKE_CURRENT_SOURCE_DIR}/data/four-steel-slabs.gdml")

  add_test(NAME "app/celer-export-geant"
    COMMAND "$<TARGET_FILE:celer-export-geant>"
    "${_geant_test_inp}" "" "test-data.root"
  )
  set_tests_properties("app/celer-export-geant" PROPERTIES
    ENVIRONMENT "${_geant_test_env}"
    REQUIRED_FILES "${_geant_test_inp}"
    LABELS "app"
  )

  add_test(NAME "app/celer-dump-data"
    COMMAND "$<TARGET_FILE:celer-dump-data>"
      "test-data.root"
  )
  set_tests_properties("app/celer-dump-data" PROPERTIES
    DEPENDS "app/celer-export-geant"
    REQUIRED_FILES "test-data.root"
    LABELS "app"
  )
endif()

#-----------------------------------------------------------------------------#
# Demo setup for HIP
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_HIP)
  set_source_files_properties(
    demo-loop/diagnostic/StepDiagnostic.cu
    demo-interactor/KNDemoKernel.cu
    demo-interactor/KNDemoKernel.thrust.cu
    PROPERTIES LANGUAGE HIP
  )
endif()

#-----------------------------------------------------------------------------#
# DEMO: physics interactions
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS)
  celeritas_add_library(celeritas_demo_interactor
    demo-interactor/LoadXs.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/XsGridParams.cc
  )
  celeritas_target_link_libraries(celeritas_demo_interactor PUBLIC
    Celeritas::celeritas
    nlohmann_json::nlohmann_json
  )

  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    add_executable(demo-interactor
      demo-interactor/demo-interactor.cc
      demo-interactor/KNDemoRunner.cc
      demo-interactor/KNDemoKernel.cu
      demo-interactor/KNDemoKernel.thrust.cu
    )
    celeritas_target_link_libraries(demo-interactor celeritas_demo_interactor)

    if(CELERITAS_BUILD_TESTS)
      set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-interactor/simple-driver.py")
      add_test(NAME "app/demo-interactor"
        COMMAND "${_python_exe}" "${_driver}"
      )
      set(_env
        "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-interactor>"
        "CELER_DISABLE_PARALLEL=1"
      )
      set_tests_properties("app/demo-interactor" PROPERTIES
        ENVIRONMENT "${_env}"
        RESOURCE_LOCK gpu
        REQUIRED_FILES "${_driver}"
        LABELS "app;nomemcheck;gpu"
        ${_disabled_unless_python}
      )
    endif()
  endif()

  # Build CPU version
  add_executable(host-demo-interactor
    demo-interactor/host-demo-interactor.cc
    demo-interactor/HostKNDemoRunner.cc
  )
  celeritas_target_link_libraries(host-demo-interactor
    Celeritas::celeritas
    celeritas_demo_interactor
  )

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-interactor/simple-driver.py")
    add_test(NAME "app/host-demo-interactor"
      COMMAND "${_python_exe}" "${_driver}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:host-demo-interactor>"
      "CELER_DISABLE_DEVICE=1"
      "CELER_DISABLE_PARALLEL=1"
      ${_omp_env}
    )
    set_tests_properties("app/host-demo-interactor" PROPERTIES
      ENVIRONMENT "${_env}"
      REQUIRED_FILES "${_driver}"
      ${_disabled_unless_python}
      ${_processors}
      LABELS "app;nomemcheck"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: geometry tracking
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA)
  add_executable(demo-rasterizer
    demo-rasterizer/demo-rasterizer.cc
    demo-rasterizer/RDemoRunner.cc
    demo-rasterizer/RDemoKernel.cu
    demo-rasterizer/ImageIO.cc
    demo-rasterizer/ImageStore.cc
  )
  set(_rasterizer_libs
    Celeritas::celeritas
    nlohmann_json::nlohmann_json
  )
  if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
    list(APPEND _rasterizer_libs VecGeom::vecgeom)
  endif()
  celeritas_target_link_libraries(demo-rasterizer ${_rasterizer_libs})

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-rasterizer/simple-driver.py")
    set(_gdml_inp "${CMAKE_CURRENT_SOURCE_DIR}/data/two-boxes.gdml")
    add_test(NAME "app/demo-rasterizer"
      COMMAND "$<TARGET_FILE:Python::Interpreter>" "${_driver}" "${_gdml_inp}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-rasterizer>"
      "CELER_DISABLE_PARALLEL=1"
    )
    set_tests_properties("app/demo-rasterizer" PROPERTIES
      ENVIRONMENT "${_env}"
      RESOURCE_LOCK gpu
      REQUIRED_FILES "${_driver};${_gdml_inp}"
      LABELS "app;nomemcheck;gpu"
    )
    if(NOT CELERITAS_CORE_GEO STREQUAL "VecGeom")
      # two-boxes.org.json isn't set up in the test directory
      set_tests_properties("app/demo-rasterizer" PROPERTIES
        DISABLED true
      )
    endif()
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: full physics loop
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS)
  set(_demo_loop_src
    demo-loop/demo-loop.cc
    demo-loop/RunnerInputIO.json.cc
    demo-loop/Runner.cc
    demo-loop/Transporter.cc
    demo-loop/diagnostic/StepDiagnostic.cc
  )

  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND _demo_loop_src
      demo-loop/diagnostic/StepDiagnostic.cu
    )
  endif()

  set(_demo_loop_libs
    Celeritas::celeritas
    nlohmann_json::nlohmann_json
    Celeritas::DeviceToolkit
  )
  if(CELERITAS_USE_CUDA AND CELERITAS_CORE_GEO STREQUAL "VecGeom")
    list(APPEND _demo_loop_libs VecGeom::vecgeom)
  endif()
  if(CELERITAS_USE_OpenMP)
    list(APPEND _demo_loop_libs OpenMP::OpenMP_CXX)
  endif()
  if(CELERITAS_USE_ROOT)
    list(APPEND _demo_loop_src demo-loop/RootOutput.cc)
    list(APPEND _demo_loop_libs ROOT::Core ROOT::Tree)
  endif()

  # Add the executable
  add_executable(demo-loop ${_demo_loop_src})
  celeritas_target_link_libraries(demo-loop ${_demo_loop_libs})

  if(NOT CELERITAS_USE_OpenMP AND
      (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
          OR CMAKE_CXX_COMPILER_ID MATCHES "Clang$"))
    celeritas_target_compile_options(demo-loop
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
    )
  endif()

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-loop/simple-driver.py")
    set(_gdml_inp "${CMAKE_CURRENT_SOURCE_DIR}/data/simple-cms.gdml")
    set(_hepmc3_inp "${CMAKE_CURRENT_SOURCE_DIR}/data/gamma-3evt-15prim.hepmc3")
    set(_mctruth_out "")
    set(_geant_exporter_env "CELER_EXPORT_GEANT_EXE=")
    if(CELERITAS_USE_ROOT)
      set(_geant_exporter_env
        "CELER_EXPORT_GEANT_EXE=$<TARGET_FILE:celer-export-geant>")
    endif()
    add_test(NAME "app/demo-loop"
      COMMAND "${_python_exe}"
      "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
      "${_geant_exporter_env}"
      "CELER_DISABLE_PARALLEL=1"
    )
    if(NOT CELERITAS_CORE_GEO STREQUAL "VecGeom")
      list(APPEND _env "CELER_DISABLE_VECGEOM=1")
    endif()
    set_tests_properties("app/demo-loop" PROPERTIES
      ENVIRONMENT "${_env};${_geant_test_env}"
      RESOURCE_LOCK gpu
      REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
      LABELS "app;nomemcheck;gpu"
    )

    # Disable test when prereqs are not available
    if(NOT (CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
       OR NOT CELERITAS_USE_Geant4
       OR NOT CELERITAS_USE_HepMC3
       OR NOT CELERITAS_USE_Python
       )
      set_tests_properties("app/demo-loop" PROPERTIES
        DISABLED true
      )
    endif()

    # Enable ROOT output for the demo-loop-cpu test when ROOT is available
    if(CELERITAS_USE_ROOT)
      set(_mctruth_out "${CMAKE_CURRENT_BINARY_DIR}/mctruth_output.root")
    endif()
    add_test(NAME "app/demo-loop-cpu"
      COMMAND "${_python_exe}"
      "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}" "${_mctruth_out}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
      "${_geant_exporter_env}"
      "CELER_LOG=debug"
      "CELER_DISABLE_DEVICE=1"
      "CELER_DISABLE_PARALLEL=1"
      ${_omp_env}
    )
    if(NOT CELERITAS_CORE_GEO STREQUAL "VecGeom")
      list(APPEND _env "CELER_DISABLE_VECGEOM=1")
    endif()
    set_tests_properties("app/demo-loop-cpu" PROPERTIES
      ENVIRONMENT "${_env};${_geant_test_env}"
      REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
      LABELS "app;nomemcheck"
      ${_processors}
    )
    if(NOT CELERITAS_USE_Geant4 OR NOT CELERITAS_USE_HepMC3
       OR NOT CELERITAS_USE_Python)
      set_tests_properties("app/demo-loop-cpu" PROPERTIES
        DISABLED true
      )
    endif()
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: Geant4 integration
#-----------------------------------------------------------------------------#
if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_Geant4)
  set(_demo_geant_integration_src
    demo-geant-integration/demo-geant-integration.cc
    demo-geant-integration/ActionInitialization.cc
    demo-geant-integration/DetectorConstruction.cc
    demo-geant-integration/EventAction.cc
    demo-geant-integration/GlobalSetup.cc
    demo-geant-integration/PrimaryGeneratorAction.cc
    demo-geant-integration/RunAction.cc
    demo-geant-integration/SensitiveDetector.cc
    demo-geant-integration/SensitiveHit.cc
    demo-geant-integration/TrackingAction.cc
  )
  set(_demo_geant_integration_libs Celeritas::accel)

  if(CELERITAS_USE_ROOT)
    function(celeritas_local_root_generate_dictionary)
      # No need for the rootmap file for an executable.
      set(CMAKE_ROOTTEST_NOROOTMAP TRUE)
      include_directories(
        # root_generate_dictionary can not handle the '$<INSTALL_INTERFACE:include>'
        # part of the Celeritas::corecel target's include, so we can not use it in
        # the 'DEPENDENCIES' property.
        "${CELERITAS_HEADER_CONFIG_DIRECTORY}"
        "${PROJECT_SOURCE_DIR}/src"
        # The include directory are attached to the Geant4 target through INTERFACE_INCLUDE_DIRECTORIES
        # which is not looked at by root_generate_dictionary.
        ${Geant4_INCLUDE_DIRS}
        ${CLHEP_INCLUDE_DIRS}
      )
      root_generate_dictionary(HitClassesRootInterfaces
        ${CMAKE_CURRENT_SOURCE_DIR}/demo-geant-integration/HitRootIO.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/demo-geant-integration/SensitiveHit.hh
        NOINSTALL
        MODULE demo-geant-integration
        # TODO Uncomment this line and remove the include_directories when
        # the minimal required ROOT version is v6.28/02 or later.
        # DEPENDENCIES ${Geant4_LIBRARIES} Celeritas::accel Celeritas::corecel
        LINKDEF "${CMAKE_CURRENT_SOURCE_DIR}/demo-geant-integration/HitClassesLinkDef.h"
      )
    endfunction(celeritas_local_root_generate_dictionary)
    celeritas_local_root_generate_dictionary()
    list(APPEND _demo_geant_integration_src
      "${CMAKE_CURRENT_BINARY_DIR}/HitClassesRootInterfaces.cxx"
      demo-geant-integration/HitRootIO.cc
    )
    list(APPEND _demo_geant_integration_libs ROOT::Tree)
  endif()

  add_executable(demo-geant-integration ${_demo_geant_integration_src})
  celeritas_target_link_libraries(demo-geant-integration
    ${_demo_geant_integration_libs}
  )
  if(CELERITAS_BUILD_TESTS AND CELERITAS_USE_HepMC3)
    if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
      if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(DEMO_MAX_NUM_TRACKS 524288)
        set(DEMO_INIT_CAPACITY 4194304)
      else()
        # Use smaller number of tracks when running on CPU
        set(DEMO_MAX_NUM_TRACKS 2048)
        set(DEMO_INIT_CAPACITY 65536)
      endif()
    else()
      # Use smaller number of tracks when running on CPU
      set(DEMO_MAX_NUM_TRACKS 1024)
      set(DEMO_INIT_CAPACITY 32768)
    endif()

    if(NOT CELERITAS_DEBUG)
      set(DEMO_EVENTS
        "${CMAKE_CURRENT_SOURCE_DIR}/data/ttbarsplit-12evt-1108prim.hepmc3")
    else()
      # Use fewer events for debug
      set(DEMO_EVENTS
        "${CMAKE_CURRENT_SOURCE_DIR}/data/ttbarbits-12evt-118prim.hepmc3")
    endif()
    if(CELERITAS_USE_ROOT)
      set(DEMO_ROOT_OPTIONS "/setup/rootBufferSize 128000
/setup/writeSDHits true")
    endif()
    configure_file(
      "demo-geant-integration/simple-cms-test.mac.in"
      "simple-cms-test.mac" @ONLY
    )
    add_test(NAME "app/demo-geant-integration"
      COMMAND "$<TARGET_FILE:demo-geant-integration>"
      "simple-cms-test.mac"
    )
    set(_env
      ${_geant_test_env}
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
    set_tests_properties("app/demo-geant-integration" PROPERTIES
      ENVIRONMENT "${_env}"
      LABELS "app"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# Utility: single-track geometry debugging
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS)
  set(_geo_check_src
    geo-check/GCheckRunner.cc
    geo-check/GCheckKernel.cc
    geo-check/geo-check.cc
  )
  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND _geo_check_src
      geo-check/GCheckKernel.cu
    )
  endif()

  set(_geo_check_libs
    Celeritas::celeritas
    nlohmann_json::nlohmann_json
    Celeritas::DeviceToolkit
  )
  if(CELERITAS_CORE_GEO STREQUAL "VecGeom")
    list(APPEND _geo_check_libs VecGeom::vecgeom)
  endif()

  add_executable(geo-check ${_geo_check_src})
  celeritas_target_link_libraries(geo-check ${_geo_check_libs})

  if(CELERITAS_BUILD_TESTS)
    set(_exe "$<TARGET_FILE:geo-check>")
    configure_file(
      "geo-check/gcheck-four-levels.json.in"
      "gcheck-four-levels.json" @ONLY
    )
    set(_json_inp "${CMAKE_CURRENT_BINARY_DIR}/gcheck-four-levels.json")
    add_test(NAME "app/geo-check"
      COMMAND "${_exe}" "${_json_inp}"
    )
    set_tests_properties("app/geo-check" PROPERTIES
      RESOURCE_LOCK gpu
      LABELS "app;gpu"
    )
    if(NOT (CELERITAS_CORE_GEO STREQUAL "VecGeom" AND CELERITAS_USE_CUDA))
      set_tests_properties("app/geo-check" PROPERTIES
        DISABLED true
      )
    endif()
  endif()
endif()
