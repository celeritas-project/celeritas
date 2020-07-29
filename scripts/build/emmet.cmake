# Celeritas options
set(CELERITAS_USE_CUDA ON CACHE BOOL "")
set(CELERITAS_USE_ROOT OFF CACHE BOOL "")
set(CELERITAS_USE_VECGEOM OFF CACHE BOOL "")

# Set rpath based on environment (loaded Spack modules); VecGeom does not
# correctly set rpath for downstream use
string(REPLACE ":" ";" _rpath "$ENV{LD_RUN_PATH}")
set(CMAKE_BUILD_RPATH "${_rpath}" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "$ENV{prefix_dir}/lib;${_rpath}" CACHE STRING "")

# Export compile commands for microsoft visual code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

# Use CUDA
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
set(CMAKE_CUDA_FLAGS "-arch=sm_35" CACHE STRING "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "")

# Use MPI
# XXX fexceptions gets added by the FindMPI script but it doesn't get escaped
# and so causes nvcc to bomb.
set(MPI_CXX_COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler=>-fexceptions;-pthread"
  CACHE STRING "")

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "")

# Enable color diagnostics when using Ninja
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic -fdiagnostics-color=always"
  CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS "-Werror all-warnings ${CMAKE_CUDA_FLAGS}"
  CACHE STRING "" FORCE)
