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
# DEMO: physics interactions
#-----------------------------------------------------------------------------#
# The CPU version still requires CUDA because some of the data parameters will do
# a copy to device (even though this isn't used by the CPU app) and that copy will
# fail in the DeviceAllocator if CUDA is disabled

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA)
  add_executable(demo-interactor
    demo-interactor/demo-interactor.cc
    demo-interactor/DetectorStore.cc
    demo-interactor/HitAllocatorStore.cc
    demo-interactor/LoadXs.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/KNDemoRunner.cc
    demo-interactor/KNDemoKernel.cu
    demo-interactor/PhysicsArrayParams.cc
    demo-interactor/detail/DetectorUtils.cu
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

  # Build CPU version
  add_executable(host-demo-interactor
    demo-interactor/LoadXs.cc
    demo-interactor/KNDemoIO.cc
    demo-interactor/PhysicsArrayParams.cc
    demo-interactor/host-demo-interactor.cc
    demo-interactor/HostDetectorStore.cc
    demo-interactor/HostKNDemoRunner.cc
  )
  target_link_libraries(host-demo-interactor celeritas
    CUDA::cudart
    nlohmann_json::nlohmann_json
  )

endif()

#-----------------------------------------------------------------------------#
# DEMO: geometry tracking
#-----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND CELERITAS_USE_CUDA AND CELERITAS_USE_VecGeom)
  # Since the demo kernel links against VecGeom, which requires CUDA separable
  # compilation, it cannot be linked directly into an executable.
  add_library(demo_rasterizer_cuda
    demo-rasterizer/RDemoRunner.cc
    demo-rasterizer/RDemoKernel.cu
  )
  set_target_properties(demo_rasterizer_cuda PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(demo_rasterizer_cuda
    PRIVATE
    celeritas
    VecGeom::vecgeomcuda
    VecGeom::vecgeomcuda_static
    nlohmann_json::nlohmann_json
  )

  # Add the executable
  add_executable(demo-rasterizer
    demo-rasterizer/demo-rasterizer.cc
    demo-rasterizer/ImageIO.cc
    demo-rasterizer/ImageStore.cc
  )
  target_link_libraries(demo-rasterizer
    celeritas
    VecGeom::vecgeom
    demo_rasterizer_cuda
    nlohmann_json::nlohmann_json
  )

  if(CELERITAS_BUILD_TESTS)
    add_test(NAME "app/demo-rasterizer"
      COMMAND "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/demo-rasterizer/simple-driver.py"
    )
    set_tests_properties("app/demo-rasterizer" PROPERTIES
      ENVIRONMENT "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-rasterizer>"
    )
  endif()
endif()

#-----------------------------------------------------------------------------#
