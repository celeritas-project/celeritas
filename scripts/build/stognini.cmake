macro(set_cache_var var type val)
  set(${var} "${val}" CACHE "${type}" "stognini.sh")
endmacro()

set_cache_var(CELERITAS_BUILD_DOCS BOOL ON)

# Dependency options
set_cache_var(CELERITAS_USE_CUDA BOOL OFF)
set_cache_var(CELERITAS_USE_Geant4 BOOL ON)
set_cache_var(CELERITAS_USE_MPI BOOL OFF)
set_cache_var(CELERITAS_USE_ROOT BOOL ON)
set_cache_var(CELERITAS_USE_SWIG_Python BOOL OFF)
set_cache_var(CELERITAS_USE_VecGeom BOOL OFF)
set_cache_var(CELERITAS_BUILD_TESTS BOOL ON)

# Build options
set_cache_var(BUILD_SHARED_LIBS BOOL ON)
set_cache_var(CMAKE_BUILD_TYPE STRING "Debug")
set_cache_var(CMAKE_SWIG_CXX_FLAGS STRING
  "-Wno-deprecated-declarations")
set_cache_var(CMAKE_CXX_FLAGS STRING
  "-Wall -Wextra -Werror -pedantic -fdiagnostics-color=always")
