#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
# GOOGLETEST
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_TESTS AND NOT TARGET GTest::GTest)
  # Externally installed gtest is not in use: pull integrated googletest
  celeritas_load_submodule(googletest)
  # Compile googletest
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/googletest.cmake)
endif()

#----------------------------------------------------------------------------#
# NLOHMANN JSON
#----------------------------------------------------------------------------#

if(CELERITAS_BUILD_DEMOS AND NOT TARGET nlohmann_json::nlohmann_json)
  # Externally installed gtest is not in use: pull integrated googletest
  celeritas_load_submodule(nlohmann_json)
  # Compile googletest
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/nlohmann_json.cmake)
endif()

#---------------------------------------------------------------------------##
