include("${CMAKE_CURRENT_LIST_DIR}/emmet.cmake")

macro(set_cache_var var type val)
  set(${var} "${val}" CACHE "${type}" "wildstyle.sh" FORCE)
endmacro()

set_cache_var(CMAKE_CUDA_ARCHITECTURES STRING "70")
set_cache_var(CMAKE_CXX_FLAGS_RELEASE STRING
  "-O3 -DNDEBUG -march=skylake-avx512 -mtune=skylake-avx512")
