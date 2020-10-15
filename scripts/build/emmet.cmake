macro(set_cache_var var type val)
  set(${var} "${val}" CACHE "${type}" "emmet.sh" FORCE)
endmacro()

# Celeritas dependency options
set_cache_var(CELERITAS_USE_CUDA BOOL ON)
set_cache_var(CELERITAS_USE_Geant4 BOOL ON)
set_cache_var(CELERITAS_USE_HepMC3 BOOL ON)
set_cache_var(CELERITAS_USE_MPI BOOL OFF)
set_cache_var(CELERITAS_USE_ROOT BOOL ON)
set_cache_var(CELERITAS_USE_VecGeom BOOL ON)
set_cache_var(CELERITAS_GIT_SUBMODULE BOOL OFF)

# Set rpath based on environment
string(REPLACE ":" ";" _rpath "$ENV{LD_RUN_PATH}")
set_cache_var(CMAKE_BUILD_RPATH STRING "${_rpath}")
set_cache_var(CMAKE_INSTALL_RPATH STRING "$ENV{prefix_dir}/lib;${_rpath}")

# Export compile commands for microsoft visual code
set_cache_var(CMAKE_EXPORT_COMPILE_COMMANDS BOOL ON)

# Build flags
set_cache_var(BUILD_SHARED_LIBS BOOL ON)
set_cache_var(CMAKE_CUDA_ARCHITECTURES STRING "35")
set_cache_var(CMAKE_CUDA_FLAGS_DEBUG STRING "-g -G")
# NOTE: CUDA error flags do not work with CMake 3.18
#set_cache_var(CMAKE_CUDA_FLAGS STRING
#  "-Werror all-warnings ${CMAKE_CUDA_FLAGS}")

set_cache_var(CELERITAS_DEBUG BOOL ON)
set_cache_var(CMAKE_BUILD_TYPE STRING "Debug")
set_cache_var(CMAKE_CXX_FLAGS STRING
  "-Wall -Wextra -Werror -pedantic -fdiagnostics-color=always")

# MPI flags
set_cache_var(MPI_CXX_SKIP_MPICXX BOOL TRUE)
if(CELERITAS_USE_MPI)
  # In CMake 3.18, these break the CUDA link line...
  set_cache_var(MPI_CXX_LINK_FLAGS STRING "")
endif()
