#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# GOOGLETEST
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_TESTS AND NOT TARGET GTest::GTest)
  include(CeleritasLoadSubmodule)

  # Externally installed gtest is not in use: pull integrated googletest
  celeritas_load_submodule(googletest)
  # Compile googletest
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/googletest.cmake)
endif()

#---------------------------------------------------------------------------##
