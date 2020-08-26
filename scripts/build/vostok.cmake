# Options
set(CELERITAS_USE_CUDA    OFF CACHE BOOL "")
set(CELERITAS_USE_ROOT    OFF CACHE BOOL "")
set(CELERITAS_USE_VecGeom ON  CACHE BOOL "")

# Libraries
string(REPLACE ":" ";" _rpath "$ENV{DYLD_FALLBACK_LIBRARY_PATH}")
set(CMAKE_BUILD_RPATH "${_rpath}" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "${_rpath};$ENV{prefix_dir}/lib" CACHE STRING "")

# Add all the warnings, and enable color diagnostics when using Ninja
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Werror -fcolor-diagnostics" CACHE STRING "")
