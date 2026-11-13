#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindVecGeom
-----------

Find the VecGeom library and set up library linking and flags for use with
Celeritas.

#]=======================================================================]

find_package(VecGeom QUIET CONFIG)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecGeom CONFIG_MODE)

if(VecGeom_FOUND AND VecGeom_CUDA_FOUND)
  # Suppress warnings from virtual function calls in RDC
  foreach(_lib VecGeom::vecgeom VecGeom::vecgeom_static)
    if(TARGET ${_lib})
      target_compile_options(${_lib}
        INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL: -Xnvlink --suppress-stack-size-warning>"
      )
      target_link_options(${_lib}
        INTERFACE "$<DEVICE_LINK:SHELL: -Xnvlink --suppress-stack-size-warning>"
      )
    endif()
  endforeach()
endif()

#-----------------------------------------------------------------------------#
