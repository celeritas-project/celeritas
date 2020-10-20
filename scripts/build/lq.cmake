macro(set_cache_var var type val)
  set(${var} "${val}" CACHE "${type}" "lq.sh")
endmacro()

# Celeritas options
set_cache_var(CELERITAS_USE_CUDA BOOL OFF)
set_cache_var(CELERITAS_USE_ROOT BOOL OFF)
set_cache_var(CELERITAS_USE_MPI  BOOL ON)
set_cache_var(CELERITAS_USE_VECGEOM BOOL ON)

# set_cache_var rpath based on environment (loaded Spack modules); VecGeom does not
# correctly set_cache_var rpath for downstream use
string(REPLACE ":" ";" _rpath "$ENV{LD_RUN_PATH}")
set_cache_var(CMAKE_BUILD_RPATH STRING "${_rpath}")
set_cache_var(CMAKE_INSTALL_RPATH STRING "$ENV{prefix_dir}/lib;${_rpath}")

# Export compile commands for microsoft visual code
set_cache_var(CMAKE_EXPORT_COMPILE_COMMANDS BOOL ON)

# Use CUDA.
set_cache_var(BUILD_SHARED_LIBS BOOL ON)
set_cache_var(CMAKE_CUDA_FLAGS STRING "-arch=sm_70")
# TODO: when using CMake 3.18, replace the above line with this one:
# set_cache_var(CMAKE_CUDA_ARCHITECTURES STRING "35")
set_cache_var(CMAKE_CUDA_FLAGS_DEBUG STRING "-g -G")
set_cache_var(CMAKE_BUILD_TYPE STRING "Debug")

# Use MPI
# XXX fexceptions gets added by the FindMPI script but it doesn't get escaped
# and so causes nvcc to bomb.
set_cache_var(MPI_CXX_COMPILE_OPTIONS STRING
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler=>-fexceptions;-pthread")

set_cache_var(MPI_CXX_SKIP_MPICXX BOOL TRUE)

# Enable color diagnostics when using Ninja
set_cache_var(CMAKE_CXX_FLAGS STRING
  "-Wall -Wextra -pedantic -fdiagnostics-color=always")
#  "-Wall -Wextra -Werror -pedantic -fdiagnostics-color=always")
set_cache_var(CMAKE_CUDA_FLAGS STRING
  "-Werror all-warnings ${CMAKE_CUDA_FLAGS}")
