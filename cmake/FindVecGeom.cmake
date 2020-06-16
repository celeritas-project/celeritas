#---------------------------------*-CMake-*----------------------------------#
# Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#[=======================================================================[.rst:

FindVecGeom
-----------

Find VecGeom and define modern CMake targets.

.. code-block:: cmake

   find_package(VecGeom REQUIRED)
   target_link_libraries(<MYTARGET> VecGeom::VecGeom)

This script changes the scope of VecGeom definitions from *global* to
*target*-based.


#]=======================================================================]

# Save compile definitions to reverse VecGeom's global add_definitions call
get_property(_SAVED_COMPILE_DEFS DIRECTORY PROPERTY COMPILE_DEFINITIONS)

find_package(VecGeom QUIET NO_MODULE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VecGeom HANDLE_COMPONENTS CONFIG_MODE)

# Restore global compile definitions
set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS "${_SAVED_COMPILE_DEFS}")

set(_VG_TARGET "VecGeom::VecGeom")
if(VECGEOM_FOUND AND NOT TARGET "${_VG_TARGET}")
  # Remove leading -D from vecgeom definitions
  foreach(_DEF IN LISTS VECGEOM_DEFINITIONS)
    string(REGEX REPLACE "^-D" "" _DEF "${_DEF}")
    list(APPEND VECGEOM_DEF_LIST "${_DEF}")
  endforeach()


  # Split libraries into "cuda" "primary" and "dependencies"
  if(VECGEOM_CUDA_STATIC_LIBRARY)
    list(GET VECGEOM_LIBRARIES -1 _VG_LIBRARY)
    if(_VG_LIBRARY MATCHES "cuda")
      list(REMOVE_AT VECGEOM_LIBRARIES -1)
    endif()
  endif()

  list(GET VECGEOM_LIBRARIES -1 _VG_LIBRARY)
  set(VECGEOM_DEP_LIBRARIES "${VECGEOM_LIBRARIES}")
  list(REMOVE_AT VECGEOM_DEP_LIBRARIES -1)

  # By default the library path has relative components (../..)
  get_filename_component(VECGEOM_LIBRARY "${_VG_LIBRARY}" REALPATH CACHE)

  set(_VG_INCLUDE_DIRS ${VECGEOM_EXTERNAL_INCLUDES})
  if(NOT VECGEOM_INCLUDE_DIR_NEXT)
    # _NEXT has been removed, or the version is too old
    set(VECGEOM_INCLUDE_DIR_NEXT "${VECGEOM_INCLUDE_DIR}")
    if(NOT IS_DIRECTORY "${VECGEOM_INCLUDE_DIR_NEXT}/VecGeom")
      message(SEND_ERROR "The installed version of VecGeom is too old")
    endif()
  endif()
  # Convert relative path to absolute and add to full list of include dirs
  get_filename_component(_VG_INCLUDE_DIR "${VECGEOM_INCLUDE_DIR_NEXT}" REALPATH CACHE)
  list(APPEND _VG_INCLUDE_DIRS "${_VG_INCLUDE_DIR}")

  if(VECGEOM_CUDA_STATIC_LIBRARY)
    # CUDA libraries aren't listed downstream of vecgeom.so, but are still needed
    list(APPEND VECGEOM_DEP_LIBRARIES "cudadevrt" "cudart")
    list(APPEND VECGEOM_DEP_LINKDIRS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()


  add_library("${_VG_TARGET}" IMPORTED UNKNOWN)
  set_target_properties("${_VG_TARGET}" PROPERTIES
    IMPORTED_LOCATION "${VECGEOM_LIBRARY}"
    INTERFACE_LINK_DIRECTORIES "${VECGEOM_DEP_LINKDIRS}"
    INTERFACE_LINK_LIBRARIES "${VECGEOM_DEP_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${_VG_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS "${VECGEOM_DEF_LIST}"
    INTERFACE_COMPILE_OPTIONS
      "$<$<COMPILE_LANGUAGE:CXX>:${VECGEOM_COMPILE_OPTIONS}>"
  )

  # VGDML may be installed
  get_filename_component(_VECGEOM_LIBRARY_DIR "${VECGEOM_LIBRARY}" DIRECTORY)
  find_library(VECGEOM_VGDML_LIBRARY vgdml
    PATHS ${_VECGEOM_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )

  if(VECGEOM_VGDML_LIBRARY)
    find_package(XercesC REQUIRED)
    add_library("VecGeom::VGDML" IMPORTED UNKNOWN)
    set_target_properties("VecGeom::VGDML" PROPERTIES
      IMPORTED_LOCATION "${VECGEOM_VGDML_LIBRARY}"
    )
    target_link_libraries(VecGeom::VGDML
      INTERFACE VecGeom::VecGeom XercesC::XercesC)
  endif()

  if(VECGEOM_CUDA_STATIC_LIBRARY)
    get_filename_component(_VG_CUDA_STATIC "${VECGEOM_CUDA_STATIC_LIBRARY}" REALPATH CACHE)
    set(_VG_CUDA_TARGET "VecGeom::Cuda")
    add_library("${_VG_CUDA_TARGET}" IMPORTED STATIC)
    set_target_properties("${_VG_CUDA_TARGET}" PROPERTIES
      IMPORTED_LOCATION "${_VG_CUDA_STATIC}"
      IMPORTED_LINK_INTERFACE_LANGUAGES CUDA
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      INTERFACE_INCLUDE_DIRECTORIES "${_VG_INCLUDE_DIRS}"
      INTERFACE_COMPILE_DEFINITIONS "${VECGEOM_DEF_LIST}"
      INTERFACE_COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CXX>:${VECGEOM_COMPILE_OPTIONS}>")
  endif()
endif()

#----------------------------------------------------------------------------#
