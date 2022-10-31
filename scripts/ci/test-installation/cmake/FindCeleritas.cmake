#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022 UT-Battelle, LLC and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindCeleritas
-------------

Find the Celeritas library.

#]=======================================================================]

find_package(Celeritas QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Celeritas CONFIG_MODE)

if(Celeritas_FOUND AND CELERITAS_USE_CUDA)
  set_target_properties(Celeritas::celeritas PROPERTIES
    CELERITAS_CUDA_LIBRARY_TYPE Shared
    CELERITAS_CUDA_STATIC_LIBRARY Celeritas::celeritas_static
    CELERITAS_CUDA_MIDDLE_LIBRARY Celeritas::celeritas
    CELERITAS_CUDA_FINAL_LIBRARY Celeritas::celeritas_final
  )
  set_target_properties(Celeritas::celeritas_static PROPERTIES
    CELERITAS_CUDA_LIBRARY_TYPE Static
    CELERITAS_CUDA_STATIC_LIBRARY Celeritas::celeritas_static
    CELERITAS_CUDA_MIDDLE_LIBRARY Celeritas::celeritas
    CELERITAS_CUDA_FINAL_LIBRARY Celeritas::celeritas_final
  )
endif()

#-----------------------------------------------------------------------------#
