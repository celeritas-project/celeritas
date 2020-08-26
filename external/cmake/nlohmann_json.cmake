#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

nlohmann_json
-------------

Include this file *directly* from the parent directory to set up JSON.

#]=======================================================================]

# Set up options
set(JSON_BuildTests OFF CACHE INTERNAL "")

add_subdirectory(nlohmann_json)

#-----------------------------------------------------------------------------#
