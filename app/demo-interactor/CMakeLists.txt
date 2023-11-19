#----------------------------------*-CMake-*----------------------------------#
# Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_HIP)
  set_source_files_properties(
    KNDemoKernel.cu
    KNDemoKernel.thrust.cu
    PROPERTIES LANGUAGE HIP
  )
endif()

celeritas_add_library(celeritas_demo_interactor
  LoadXs.cc
  KNDemoIO.cc
  XsGridParams.cc
)
celeritas_target_link_libraries(celeritas_demo_interactor PUBLIC
  Celeritas::celeritas
  nlohmann_json::nlohmann_json
)

# Build CPU version
add_executable(host-demo-interactor
  host-demo-interactor.cc
  HostKNDemoRunner.cc
)
celeritas_target_link_libraries(host-demo-interactor
  Celeritas::celeritas
  celeritas_demo_interactor
)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  # Build GPU version
  add_executable(demo-interactor
    demo-interactor.cc
    KNDemoRunner.cc
    KNDemoKernel.cu
    KNDemoKernel.thrust.cu
  )
  celeritas_target_link_libraries(demo-interactor celeritas_demo_interactor)
endif()

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

if(NOT BUILD_TESTING)
  return()
endif()

set(_driver "${CMAKE_CURRENT_SOURCE_DIR}/simple-driver.py")
add_test(NAME "app/host-demo-interactor"
  COMMAND "${CELER_PYTHON}" "${_driver}"
)
set(_env
  "CELERITAS_DEMO_EXE=$<TARGET_FILE:host-demo-interactor>"
  "CELER_DISABLE_DEVICE=1"
  "CELER_DISABLE_PARALLEL=1"
  ${CELER_OMP_ENV}
)
set_tests_properties("app/host-demo-interactor" PROPERTIES
  ENVIRONMENT "${_env}"
  REQUIRED_FILES "${_driver}"
  ${CELER_NEEDS_PYTHON}
  ${CELER_PROCESSORS}
  LABELS "app;nomemcheck"
)

if(CELERITAS_USE_CUDA OR CELERITAS_USE_HIP)
  add_test(NAME "app/demo-interactor"
    COMMAND "${CELER_PYTHON}" "${_driver}"
  )
  set(_env
    "CELERITAS_DEMO_EXE=$<TARGET_FILE:demo-interactor>"
    "CELER_DISABLE_PARALLEL=1"
  )
  set(_props ${CELER_PROCESSORS} ${CELER_NEEDS_PYTHON})
  if(CELER_DISABLE_DEVICE)
    list(APPEND _props DISABLED true)
  endif()
  set_tests_properties("app/demo-interactor" PROPERTIES
    ENVIRONMENT "${_env}"
    RESOURCE_LOCK gpu
    REQUIRED_FILES "${_driver}"
    LABELS "app;nomemcheck;gpu"
    ${_props}
  )
endif()

#-----------------------------------------------------------------------------#
