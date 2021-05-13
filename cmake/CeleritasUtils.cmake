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

function(celeritas_link_vecgeom_cuda target)
return()

#  Readd when using
  #set_target_properties(${target} PROPERTIES
  #  LINKER_LANGUAGE CUDA
  #  CUDA_SEPARABLE_COMPILATION ON
  #)

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
  if(BUILD_SHARED_LIBS)
  target_link_libraries(${target}_static
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    ${PRIVATE_DEPS}
  )
  set_target_properties(${target}_static
    PROPERTIES LINKER_LANGUAGE CXX
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
endif()

endfunction()


define_property(TARGET PROPERTY CELERITAS_CUDA_LIBRARY_TYPE
  BRIEF_DOCS "Indicate the type of cuda library (STATIC and SHARED for nvlink usage, FINAL for linking into not cuda library/executable"
  FULL_DOCS "Indicate the type of cuda library (STATIC and SHARED for nvlink usage, FINAL for linking into not cuda library/executable"
)
define_property(TARGET PROPERTY CELERITAS_CUDA_FINAL_LIBRARY
  BRIEF_DOCS "Name of the final library corresponding to this cuda library"
  FULL_DOCS "Name of the final library corresponding to this cuda library"
)
define_property(TARGET PROPERTY CELERITAS_CUDA_STATIC_LIBRARY
  BRIEF_DOCS "Name of the static library corresponding to this cuda library"
  FULL_DOCS "Name of the static library corresponding to this cuda library"
)
define_property(TARGET PROPERTY CELERITAS_CUDA_MIDDLE_LIBRARY
  BRIEF_DOCS "Name of the shared (without nvlink step) library corresponding to this cuda library"
  FULL_DOCS "Name of the shared (without nvlink step) library corresponding to this cuda library"
)

function(celeritas_add_library target)
  list(SUBLIST ARGV 1 -1 NEWARGV)

  if(NOT BUILD_SHARED_LIBS OR NOT CELERITAS_USE_CUDA)
    add_library(${target} ${NEWARGV})
    add_library(${target}_final ALIAS ${target})
    add_library(${target}_cuda ALIAS ${target})
    add_library(${target}_static ALIAS ${target})
    return()
  endif()

  add_library(${target}_objects OBJECT ${NEWARGV})
  add_library(${target}_static STATIC $<TARGET_OBJECTS:${target}_objects>)
  add_library(${target}_cuda SHARED $<TARGET_OBJECTS:${target}_objects>)
  # We need to use a dummy file as a library (per cmake) needs to contains
  # at least one source file.  The real content of the library will be
  # the cmake_device_link.o resulting from the execution of `nvcc -dlink`
  # Also non-cuda related test, for example `gtest_detail_Macros`,
  # will need to be linked again libceleritas_final while a library
  # that the detends on and that uses Celeritas::Core (for example
  # libCeleritasTest.so) will need to be linked against `libceleritas_cuda`.
  # If both the `_cuda` and `_final` contains the `.o` files we would
  # then have duplicated symbols (Here the symptoms will a crash
  # during the cuda library initialization rather than a link error).
  add_library(${target}_final SHARED ../src/base/dummy.cu)

  set_target_properties(${target}_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
  )

  set_target_properties(${target}_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON # We really don't want nvlink called.
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF # We really don't want nvlink called.
    CELERITAS_CUDA_LIBRARY_TYPE Shared
    CELERITAS_CUDA_FINAL_LIBRARY ${target}_final
    CELERITAS_CUDA_STATIC_LIBRARY ${target}_static
  )

  set_target_properties(${target}_static PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    # CUDA_RESOLVE_DEVICE_SYMBOLS OFF # Default for static library
    CELERITAS_CUDA_LIBRARY_TYPE Static
    CELERITAS_CUDA_FINAL_LIBRARY ${target}_final
    CELERITAS_CUDA_MIDDLE_LIBRARY ${target}_cuda
  )

  set_target_properties(${target}_final PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON # Default for shared library
    CELERITAS_CUDA_LIBRARY_TYPE Final
    CELERITAS_CUDA_STATIC_LIBRARY ${target}_static
    CELERITAS_CUDA_MIDDLE_LIBRARY ${target}_cuda
  )

  if (CELERITAS_USE_VecGeom)
    target_link_libraries(${target}_cuda
      PRIVATE VecGeom::vecgeom
    )
    target_link_libraries(${target}_final
      PRIVATE VecGeom::vecgeom
    )
    target_link_libraries(${target}_final
      PRIVATE VecGeom::vecgeomcuda
    )
  endif()
  target_link_libraries(${target}_final
    PUBLIC ${target}_cuda
  )

  target_link_options(${target}_final
    PRIVATE
    $<DEVICE_LINK:$<TARGET_FILE:celeritas_static>>
    $<DEVICE_LINK:$<TARGET_FILE:${target}_static>>
  )
  if (CELERITAS_USE_VecGeom)
    get_property(vecgeom_static_target_location TARGET VecGeom::vecgeomcuda_static PROPERTY LOCATION)
    target_link_options(${target}_final
      PRIVATE
      $<DEVICE_LINK:${vecgeom_static_target_location}>
    )
    target_link_libraries(${target}_objects
      PRIVATE VecGeom::vecgeom
    )
    target_link_libraries(${target}_objects
      PRIVATE VecGeom::vecgeomcuda_static
    )
  endif()

  add_dependencies(${target}_final ${target}_cuda)
  add_dependencies(${target}_final ${target}_static)

  add_library(${target} ALIAS ${target}_final)

endfunction()
#-----------------------------------------------------------------------------#
