macro(set_cache_var var type val)
  set(${var} "${val}" CACHE "${type}" "yuri.sh")
endmacro()

set_cache_var(CELERITAS_BUILD_DOCS BOOL ON)

# Dependency options
set_cache_var(CELERITAS_USE_CUDA BOOL OFF)
set_cache_var(CELERITAS_USE_Geant4 BOOL OFF)
set_cache_var(CELERITAS_GIT_SUBMODULE BOOL OFF)
set_cache_var(CELERITAS_USE_MPI BOOL ON)
set_cache_var(CELERITAS_USE_ROOT BOOL ON)
set_cache_var(CELERITAS_USE_SWIG_Python BOOL ON)
set_cache_var(CELERITAS_USE_VecGeom BOOL OFF)

# Build options
set_cache_var(BUILD_SHARED_LIBS BOOL ON)
set_cache_var(CMAKE_BUILD_TYPE STRING "Debug")
set_cache_var(CMAKE_CXX_FLAGS STRING
  " -fdiagnostics-color=always")
set_cache_var(CMAKE_SWIG_CXX_FLAGS STRING
  "-Wno-deprecated-declarations")
set_cache_var(CMAKE_CXX_FLAGS STRING
  " -fdiagnostics-color=always")
