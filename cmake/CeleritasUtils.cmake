#----------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

CeleritasUtils
--------------

CMake utility functions for Celeritas.

.. command:: celeritas_find_package_config

  ::

    celeritas_find_package_config(<package> [...])

  Find the given package specified by a config file, but print location and
  version information while loading. A well-behaved package Config.cmake file
  should already include this, but several of the HEP packages (ROOT, Geant4,
  VecGeom do not, so this helps debug system configuration issues.

  The "Found" message should only display the first time a package is found and
  should be silent on subsequent CMake reconfigures.

  Once upstream packages are updated, this can be replaced by ``find_package``.


.. command:: celeritas_link_vecgeom_cuda

  Link the given target privately against VecGeom with CUDA support.

  ::

    celeritas_link_vecgeom_cuda(<target>)

#]=======================================================================]
include(FindPackageHandleStandardArgs)

macro(celeritas_find_package_config _package)
  find_package(${_package} CONFIG ${ARGN})
  find_package_handle_standard_args(${_package} CONFIG_MODE)
endmacro()

macro(celeritas_add_library)
  if(CELERITAS_USE_CUDA)
    if(BUILD_SHARED_LIBS)
      cuda_add_library(${ARGV} SHARED)
    else()
      cuda_add_library(${ARGV} STATIC)
    endif()
else()
     add_library(${ARGV})
  endif()
endmacro()

function(celeritas_link_vecgeom_cuda target)

#  Readd when using
#  set_target_properties(${target} PROPERTIES
#    LINKER_LANGUAGE CUDA
#    CUDA_SEPARABLE_COMPILATION ON
#  )

  # Note: the repeat (target name, location) below is due the author of  cuda_add_library_depend
  # not knowing how to automatically go from the target to the real file from a generator expression in add_custom_command
  get_property(vecgeom_static_target_location TARGET VecGeom::vecgeomcuda_static PROPERTY LOCATION)
  cuda_add_library_depend(${target} VecGeom::vecgeom VecGeom::vecgeomcuda_static ${vecgeom_static_target_location})
  target_link_libraries(${target}
    PRIVATE
    VecGeom::vecgeom
    VecGeom::vecgeomcuda
    VecGeom::vecgeomcuda_static
  )
  target_link_libraries(${target}_static
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    ${PRIVATE_DEPS}
  )
  set_target_properties(${target}_static
    PROPERTIES LINKER_LANGUAGE CXX
  )

  get_target_property(target_include_directories ${target}
    INCLUDE_DIRECTORIES
  )
  set_target_properties(${target}_static PROPERTIES
    INCLUDE_DIRECTORIES "${target_include_directories}"
  )

  get_target_property(target_interface_include_directories ${target}
    INTERFACE_INCLUDE_DIRECTORIES
  )
  set_target_properties(${target}_static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${target_interface_include_directories}")

  get_target_property(target_LINK_LIBRARIES ${target}
    LINK_LIBRARIES
  )
  get_target_property(target_INTERFACE_LINK_LIBRARIES ${target}
    INTERFACE_LINK_LIBRARIES )
  set_target_properties(${target}_static PROPERTIES
    LINK_LIBRARIES "${target_LINK_LIBRARIES}"
    INTERFACE_LINK_LIBRARIES "${target_INTERFACE_LINK_LIBRARIES}"
  )

  set_target_properties(${target}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )

endfunction()

function(celeritas_link_vecgeom_cuda_dep target)
  celeritas_link_vecgeom_cuda(${ARGV})

  # Note: the repeat (target name, location) below is due the author of  cuda_add_library_depend
  # not knowing how to automatically go from the target to the real file from a generator expression in add_custom_command
  get_property(vecgeom_static_target_location TARGET VecGeom::vecgeomcuda_static PROPERTY LOCATION)
  cuda_add_library_depend(${target} celeritas_static ../src/libceleritas_static.a VecGeom::vecgeomcuda_static ${vecgeom_static_target_location})
endfunction()
#-----------------------------------------------------------------------------#
