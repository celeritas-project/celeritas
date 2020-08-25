#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4 AND CELERITAS_USE_ROOT)
  # Exporter
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

  # Reader
  add_executable(geant-exporter-cat geant-exporter/geant-exporter-cat.cc)
  target_link_libraries(geant-exporter-cat
    Celeritas::Core
    Celeritas::IO
  )

  if(CELERITAS_BUILD_TESTS)
    set(_geant_test_inp
      "${CMAKE_CURRENT_SOURCE_DIR}/geant-exporter/data/two-boxes.gdml"
    )
    add_test(NAME "app/geant-exporter"
      COMMAND "$<TARGET_FILE:geant-exporter>"
      "${_geant_test_inp}" "test-data.root"
    )

    # Set environment variables from Geant4-exported configuration
    set(_geant_test_env)
    foreach(_ds IN LISTS Geant4_DATASETS)
      list(APPEND _geant_test_env
        "${Geant4_DATASET_${_ds}_ENVVAR}=${Geant4_DATASET_${_ds}_PATH}")
    endforeach()
    set_tests_properties("app/geant-exporter" PROPERTIES
      ENVIRONMENT "${_geant_test_env}"
      REQUIRED_FILES "${_geant_test_inp}"
    )

    add_test(NAME "app/geant-exporter-cat"
      COMMAND "$<TARGET_FILE:geant-exporter-cat>"
        "test-data.root"
    )
    set_tests_properties("app/geant-exporter-cat" PROPERTIES
      DEPENDS "app/geant-exporter"
      REQUIRED_FILES "test-data.root"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA)
  add_executable(demo-interactor
    demo-interactor/demo-interactor.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/KNDemoRunner.cc
    demo-interactor/KNDemoKernel.cu
  )
  target_link_libraries(demo-interactor celeritas
    CUDA::cudart
    nlohmann_json::nlohmann_json
  )

  if(CELERITAS_BUILD_TESTS)
    add_test(NAME "app/demo-interactor"
      COMMAND "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/demo-interactor/simple-driver.py"
    )
    set_tests_properties("app/demo-interactor" PROPERTIES
      ENVIRONMENT "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-interactor>"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
