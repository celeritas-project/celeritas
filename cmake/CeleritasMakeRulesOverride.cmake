#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasMakeRulesOverride
--------------------------

Override CMake platform information with defaults. This is the cleanest way to
change default flags in a safe (platform-independent and non-user-ignoring) way.
See
https://stackoverflow.com/questions/28732209/change-default-value-of-cmake-cxx-flags-debug-and-friends-in-cmake
and
https://cmake.org/cmake/help/latest/variable/CMAKE_USER_MAKE_RULES_OVERRIDE.html

#]=======================================================================]

# Enable lots of warnings for GCC and Clang by default
foreach(_lang C CXX)
  set(_id "${CMAKE_${_lang}_COMPILER_ID}")
  if(_id STREQUAL "GNU" OR _id MATCHES "Clang$")
    string(APPEND CMAKE_${_lang}_FLAGS_INIT "-Wall -Wextra -pedantic")
  endif()
endforeach()

if("$ENV{TERM}" MATCHES "xterm")
  foreach(_lang C CXX)
    set(_id "${CMAKE_${_lang}_COMPILER_ID}")
    if(_id STREQUAL "GNU")
      string(APPEND CMAKE_${_lang}_FLAGS_INIT " -fdiagnostics-color=always")
    elseif(_id MATCHES "Clang$")
      string(APPEND CMAKE_${_lang}_FLAGS_INIT " -fcolor-diagnostics")
    endif()
  endforeach()
endif()

unset(_id)
unset(_lang)

#-----------------------------------------------------------------------------#
