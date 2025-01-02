#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindVecGeom
-----------

Find the VecGeom library and set up library linking and flags for use with
Celeritas.

#]=======================================================================]

# TODO: remove once we require a veccore version including https://github.com/root-project/veccore/commit/743566fac1e9b2eaeb0f0b63242442ba430e0cc0
cmake_policy(PUSH)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(VecGeom QUIET CONFIG)
cmake_policy(POP)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecGeom CONFIG_MODE)

if(VecGeom_FOUND AND TARGET VecGeom::vecgeomcuda)
  get_target_property(_vecgeom_lib_type VecGeom::vecgeom TYPE)
  if (_vecgeom_lib_type STREQUAL "STATIC_LIBRARY")
     set(_vecgeom_cuda_runtime "Static")
     set(_vecgeom_middle_library_suffix "_static")
  else()
     set(_vecgeom_cuda_runtime "Shared")
     set(_vecgeom_middle_library_suffix "")
  endif()
  get_target_property(_vecgeom_lib_rdc_final VecGeom::vecgeomcuda CUDA_RDC_FINAL_LIBRARY)
  if(NOT _vecgeom_lib_rdc_final)
    set_target_properties(VecGeom::vecgeom PROPERTIES
      CUDA_RDC_STATIC_LIBRARY VecGeom::vecgeomcuda_static
      CUDA_RDC_MIDDLE_LIBRARY VecGeom::vecgeomcuda${_vecgeom_middle_library_suffix}
      CUDA_RDC_FINAL_LIBRARY VecGeom::vecgeomcuda
    )
    set_target_properties(VecGeom::vecgeomcuda PROPERTIES
      CUDA_RDC_LIBRARY_TYPE Shared
      #CUDA_RUNTIME_LIBRARY ${_vecgeom_cuda_runtime}
    )
    set_target_properties(VecGeom::vecgeomcuda_static PROPERTIES
      CUDA_RDC_LIBRARY_TYPE Static
    )
  endif()
  # Suppress warnings from virtual function calls in RDC
  foreach(_lib VecGeom::vecgeomcuda VecGeom::vecgeomcuda_static)
    target_compile_options(${_lib}
      INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL: -Xnvlink --suppress-stack-size-warning>"
    )
    target_link_options(${_lib}
      INTERFACE "$<DEVICE_LINK:SHELL: -Xnvlink --suppress-stack-size-warning>"
    )
  endforeach()

  if(NOT _vecgeom_lib_rdc_final)
    # Inform cuda_rdc_add_library code
    foreach(_lib VecGeom::vecgeom VecGeom::vecgeomcuda
        VecGeom::vecgeomcuda_static)
      set_target_properties(${_lib} PROPERTIES
        CUDA_RDC_STATIC_LIBRARY VecGeom::vecgeomcuda_static
        CUDA_RDC_MIDDLE_LIBRARY VecGeom::vecgeomcuda${_vecgeom_middle_library_suffix}
        CUDA_RDC_FINAL_LIBRARY VecGeom::vecgeomcuda
      )
    endforeach()
  endif()
endif()

#-----------------------------------------------------------------------------#
