#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# INTERFACE LIBRARIES
#----------------------------------------------------------------------------#

# Add a library for propagating common build flags and Celeritas includes
celeritas_add_src_library(BuildFlags INTERFACE)

# Require at least C++17
target_compile_features(BuildFlags INTERFACE cxx_std_17)
if(CELERITAS_USE_CUDA)
  target_compile_features(BuildFlags INTERFACE cuda_std_17)
endif()
if(WIN32)
  target_compile_definitions(BuildFlags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:NOMINMAX NOGDI>
  )

  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # __cplusplus: use standard-compliant C++ version macro
    # preprocessor: use standard-compliant preprocessor
    # EHsc: enable standard C++ exception handling
    target_compile_options(BuildFlags INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus /Zc:preprocessor /EHsc>
    )
  endif()

  # Inheriting via dominance is correct behavior
  target_compile_options(BuildFlags
    INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/wd4267$<SEMICOLON>/wd4250>"
  )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  # GCC 13 and 14 have *lots* of false positives, and not just on our project
  # see: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107532
  celeritas_target_compile_options(BuildFlags
    INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-dangling-reference>"
  )
endif()

celeritas_target_include_directories(BuildFlags
  INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CELERITAS_HEADER_CONFIG_DIRECTORY}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

#----------------------------------------------------------------------------#

# Define an interface library for propagating include paths for GPU compilers.
# This allow API calls, so long as the compiler or linker implicitly bring in
# the correct CUDA/ROCM libraries. Those libraries are *not* explicitly linked
# because mistakenly linking against both cudart and cudart_static (one of which
# might come from upstream libraries such as vecgeom) can cause nasty link- and
# run-time errors.
celeritas_add_interface_library(ExtDeviceApi)

if(CELERITAS_USE_CUDA)
  target_link_libraries(ExtDeviceApi INTERFACE CUDA::toolkit)

  # NVTX requires `-ldl`
  # This does not appears to be needed anymore with
  # CMake 3.26.5 and CUDA sdk 12.6.
  target_link_libraries(ExtDeviceApi INTERFACE ${CMAKE_DL_LIBS})
elseif(CELERITAS_USE_HIP)
  if(CMAKE_HIP_COMPILER_ROCM_ROOT)
    # Undocumented CMake variable
    set(ROCM_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
  else()
    # This works on some older OLCF environment
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm headers")
  endif()
  target_include_directories(ExtDeviceApi
    SYSTEM INTERFACE "${ROCM_PATH}/include"
  )
  if(ROCTX_LIBRARY)
    target_link_libraries(ExtDeviceApi INTERFACE "${ROCTX_LIBRARY}")
  endif()
  if(NOT BUILD_SHARED_LIBS)
    # Downstream libs don't link against correct HIP dependencies when using
    # static libraries, and even though the code below propagates the library
    # names (-lamdhip64) CMake fails to include the link directories
    # (/opt/rocm/lib)
    target_link_libraries(ExtDeviceApi
      INTERFACE "$<INSTALL_INTERFACE:${CMAKE_HIP_IMPLICIT_LINK_LIBRARIES}>"
    )
    target_link_directories(ExtDeviceApi
      INTERFACE "$<INSTALL_INTERFACE:${CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES}>"
    )
  endif()
endif()

#----------------------------------------------------------------------------#

# Add libraries for optional external dependencies
macro(celer_add_ext_library pkg)
  celeritas_add_interface_library(Ext${pkg})

  if(CELERITAS_USE_${pkg})
    target_link_libraries(Ext${pkg} INTERFACE ${ARGN})
  endif()
endmacro()

celer_add_ext_library(MPI MPI::MPI_CXX)
celer_add_ext_library(OpenMP OpenMP::OpenMP_CXX)
celer_add_ext_library(VecGeom VecGeom::vecgeom)
if(CELERITAS_USE_CUDA)
  # VecGeom headers propagate CUDA includes into host compiler code
  # see https://gitlab.cern.ch/VecGeom/VecGeom/-/merge_requests/1289
  target_link_libraries(ExtVecGeom INTERFACE Celeritas::ExtDeviceApi)
endif()

celer_add_ext_library(Perfetto)
if(CELERITAS_BUILTIN_Perfetto)
  target_link_libraries(ExtPerfetto INTERFACE Celeritas::perfetto)
endif()

celer_add_ext_library(Thrust)
if(CELERITAS_USE_CUDA)
  # Perlmutter with RDC enabled complains about missing dlclose
  target_link_libraries(ExtThrust INTERFACE ${CMAKE_DL_LIBS})
endif()

celeritas_add_interface_library(ExtGeant4Geo)
if(CELERITAS_USE_Geant4)
  celeritas_get_g4libs(_cg4_libs geometry)
  target_link_libraries(ExtGeant4Geo INTERFACE ${_cg4_libs})
endif()

#----------------------------------------------------------------------------#
# CODE LIBRARIES
#----------------------------------------------------------------------------#

# Add individual libraries
add_subdirectory(corecel)
add_subdirectory(geocel)
add_subdirectory(orange)
add_subdirectory(celeritas)
if(CELERITAS_USE_Geant4)
  add_subdirectory(accel)
endif()

# Add alias as a bridge library: see cmake/CeleritasConfig.cmake.in
# TODO: move to a separate library for just high level interfaces
if(CELERITAS_USE_Geant4)
  add_library(Celeritas::G4 ALIAS accel)
else()
  add_library(Celeritas::G4 ALIAS BuildFlags)
endif()

# Install header files
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT development
  FILES_MATCHING REGEX ".*\\.hh?$"
)

#---------------------------------------------------------------------------##
