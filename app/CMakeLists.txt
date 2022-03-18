#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
endif()

#-----------------------------------------------------------------------------#
# Geant4/ROOT data interface
#-----------------------------------------------------------------------------#

# Exporter
add_executable(celer-export-geant celer-export-geant.cc)
celeritas_target_link_libraries(celer-export-geant
  Celeritas::Core
  Celeritas::IO
)

# Reader
add_executable(celer-dump-data celer-dump-data.cc)
celeritas_target_link_libraries(celer-dump-data
  Celeritas::Core
  Celeritas::IO
)

if(CELERITAS_USE_ROOT AND CELERITAS_USE_Geant4 AND CELERITAS_BUILD_TESTS)
  set(_geant_test_inp "${CMAKE_CURRENT_SOURCE_DIR}/data/four-steel-slabs.gdml")

  add_test(NAME "app/celer-export-geant"
    COMMAND "$<TARGET_FILE:celer-export-geant>"
    "${_geant_test_inp}" "test-data.root"
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
    demo-loop/generated/AlongAndPostStepKernel.cu
    demo-loop/generated/ProcessInteractionsKernel.cu
    demo-loop/generated/PreStepKernel.cu
    demo-loop/generated/CleanupKernel.cu
    demo-loop/diagnostic/TrackDiagnostic.cu
    demo-loop/diagnostic/StepDiagnostic.cu
    demo-loop/diagnostic/ParticleProcessDiagnostic.cu
    demo-loop/diagnostic/EnergyDiagnostic.cu
    demo-interactor/KNDemoKernel.cu
    demo-interactor/KNDemoKernel.thrust.cu
    PROPERTIES LANGUAGE HIP
  )
endif()

#-----------------------------------------------------------------------------#
# DEMO: physics interactions
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS)
  add_library(celeritas_demo_interactor
    demo-interactor/LoadXs.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/XsGridParams.cc
  )
  celeritas_target_link_libraries(celeritas_demo_interactor PUBLIC
    celeritas
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
    celeritas
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
    )
    set_tests_properties("app/host-demo-interactor" PROPERTIES
      ENVIRONMENT "${_env}"
      REQUIRED_FILES "${_driver}"
      ${_disabled_unless_python}
      LABELS "app;nomemcheck"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: geometry tracking
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  add_executable(demo-rasterizer
    demo-rasterizer/demo-rasterizer.cc
    demo-rasterizer/RDemoRunner.cc
    demo-rasterizer/RDemoKernel.cu
    demo-rasterizer/ImageIO.cc
    demo-rasterizer/ImageStore.cc
  )
  celeritas_target_link_libraries(demo-rasterizer
    Celeritas::Core
    nlohmann_json::nlohmann_json
    VecGeom::vecgeom
  )

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-rasterizer/simple-driver.py")
    set(_gdml_inp "${CMAKE_CURRENT_SOURCE_DIR}/demo-rasterizer/two-boxes.gdml")
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
  endif()
endif()

#-----------------------------------------------------------------------------#
# DEMO: full physics loop
#-----------------------------------------------------------------------------#
set(_demo_loop_gen)
macro(celeritas_gen_demo_loop_kernel class func threads)
  celeritas_gen(_demo_loop_gen
    "gen-demo-loop-kernel.py" "demo-loop/generated/${class}Kernel"
    --class ${class}  --func ${func} --threads ${threads}
  )
endmacro()

# Generate demo loop kernel/host loop code for host/device
celeritas_gen_demo_loop_kernel("AlongAndPostStep" "along_and_post_step"
  "'states.size()'")
celeritas_gen_demo_loop_kernel("Cleanup" "cleanup" "1")
celeritas_gen_demo_loop_kernel("PreStep" "pre_step" "'states.size()'")
celeritas_gen_demo_loop_kernel("ProcessInteractions" "process_interactions"
  "'states.size()'")

if(CELERITAS_BUILD_DEMOS)
  set(_demo_loop_src
    demo-loop/demo-loop.cc
    demo-loop/LDemoIO.cc
    demo-loop/Transporter.cc
    demo-loop/diagnostic/EnergyDiagnostic.cc
    demo-loop/diagnostic/ParticleProcessDiagnostic.cc
    demo-loop/diagnostic/StepDiagnostic.cc
    demo-loop/diagnostic/TrackDiagnostic.cc
    ${_demo_loop_gen}
  )
  set(_demo_loop_libs)

  if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
    list(APPEND _demo_loop_src
      demo-loop/diagnostic/EnergyDiagnostic.cu
      demo-loop/diagnostic/ParticleProcessDiagnostic.cu
      demo-loop/diagnostic/StepDiagnostic.cu
      demo-loop/diagnostic/TrackDiagnostic.cu
      )
  endif()

  set(_demo_loop_libs
    Celeritas::Core
    nlohmann_json::nlohmann_json
    Celeritas::IO
  )
  if(CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
    list(APPEND _demo_loop_libs VecGeom::vecgeom)
  endif()
  if(CELERITAS_USE_OpenMP)
    list(APPEND _demo_loop_libs OpenMP::OpenMP_CXX)
  endif()

  # Add the executable
  add_executable(demo-loop ${_demo_loop_src})
  celeritas_target_link_libraries(demo-loop ${_demo_loop_libs})

  if(NOT CELERITAS_USE_OpenMP)
    celeritas_target_compile_options(demo-loop
      PRIVATE -Wno-unknown-pragmas)
  endif()

  if(CELERITAS_BUILD_TESTS)
    set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/demo-loop/simple-driver.py")
    set(_gdml_inp "${PROJECT_SOURCE_DIR}/app/data/simple-cms.gdml")
    set(_hepmc3_inp "${PROJECT_SOURCE_DIR}/app/data/input.hepmc3")
    if(TARGET celer-export-geant)
      set(_geant_exporter_env
        "CELERITAS_GEANT_EXPORTER_EXE=$<TARGET_FILE:celer-export-geant>")
    endif()
    add_test(NAME "app/demo-loop"
      COMMAND "${_python_exe}"
      "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
      "${_geant_exporter_env}"
      "CELER_DISABLE_PARALLEL=1"
    )
    if(NOT CELERITAS_USE_VecGeom)
      list(APPEND _env "CELER_DISABLE_VECGEOM=1")
    endif()
    set_tests_properties("app/demo-loop" PROPERTIES
      ENVIRONMENT "${_env};${_geant_test_env}"
      RESOURCE_LOCK gpu
      REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
      LABELS "app;nomemcheck;gpu"
    )

    # Disable test when prereqs are not available or for debug builds
    # (debug failures seem limited to older architectures with debug assertions
    # enabled)
    if(NOT (CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
       OR NOT CELERITAS_USE_Geant4
       OR NOT CELERITAS_USE_ROOT
       OR NOT CELERITAS_USE_HepMC3
       OR NOT CELERITAS_USE_Python
       OR (CELERITAS_USE_VecGeom AND CELERITAS_DEBUG
           AND CMAKE_BUILD_TYPE STREQUAL "Debug")
       )
      set_tests_properties("app/demo-loop" PROPERTIES
        DISABLED true
      )
    endif()

    add_test(NAME "app/demo-loop-cpu"
      COMMAND "${_python_exe}"
      "${_driver}" "${_gdml_inp}" "${_hepmc3_inp}"
    )
    set(_env
      "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-loop>"
      "${_geant_exporter_env}"
      "CELER_DISABLE_DEVICE=1"
      "CELER_DISABLE_PARALLEL=1"
    )
    if(NOT CELERITAS_USE_VecGeom)
      list(APPEND _env "CELER_DISABLE_VECGEOM=1")
    endif()
    set_tests_properties("app/demo-loop-cpu" PROPERTIES
      ENVIRONMENT "${_env};${_geant_test_env}"
      REQUIRED_FILES "${_driver};${_gdml_inp};${_hepmc3_inp}"
      LABELS "app;nomemcheck"
    )
    if(NOT (CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
       OR NOT CELERITAS_USE_Geant4
       OR NOT CELERITAS_USE_ROOT
       OR NOT CELERITAS_USE_HepMC3)
      set_tests_properties("app/demo-loop-cpu" PROPERTIES
        DISABLED true
      )
    endif()
  endif()
endif()

#-----------------------------------------------------------------------------#
