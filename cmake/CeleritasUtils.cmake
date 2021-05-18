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
celeritas_target_link_libraries(${target} PRIVATE VecGeom::vecgeom)
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
  get_target_property(_target_interface_link_libraries ${target}
    INTERFACE_LINK_LIBRARIES )
  set_target_properties(${target}_static PROPERTIES
    LINK_LIBRARIES "${target_LINK_LIBRARIES}"
    INTERFACE_LINK_LIBRARIES "${_target_interface_link_libraries}"
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
define_property(TARGET PROPERTY CELERITAS_CUDA_OBJECT_LIBRARY
  BRIEF_DOCS "Name of the object (without nvlink step) library corresponding to this cuda library"
  FULL_DOCS "Name of the object (without nvlink step) library corresponding to this cuda library"
)

function(celeritas_sources_contains_cuda OUTPUT_VARIABLE)
  set(_contains_cuda FALSE)
  foreach(_source ${ARGN})
    get_source_file_property(_iscudafile ${_source} LANGUAGE)
    if(_iscudafile)
      if (${_iscudafile} STREQUAL "CUDA")
        set(_contains_cuda TRUE)
      endif()
    else()
      get_filename_component(_ext "${_source}" LAST_EXT)
      if(_ext STREQUAL ".cu")
        set(_contains_cuda TRUE)
        break()
      endif()
    endif()
  endforeach()
  set(${OUTPUT_VARIABLE} ${_contains_cuda} PARENT_SCOPE)
endfunction()

function(celeritas_lib_contains_cuda OUTPUT_VARIABLE target)
  celeritas_strip_alias(target ${target})
  get_target_property(_target_sources ${target} SOURCES)

  celeritas_sources_contains_cuda(_contains_cuda ${_target_sources})
  set(${OUTPUT_VARIABLE} ${_contains_cuda} PARENT_SCOPE)
endfunction()

function(celeritas_add_library target)

  celeritas_sources_contains_cuda(_contains_cuda ${ARGN})

  if(NOT BUILD_SHARED_LIBS OR NOT CELERITAS_USE_CUDA OR NOT _contains_cuda)
    add_library(${target} ${ARGN})
    return()
  endif()

  add_library(${target}_objects OBJECT ${ARGN})
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
    CELERITAS_CUDA_MIDDLE_LIBRARY ${target}_cuda
    CELERITAS_CUDA_STATIC_LIBRARY ${target}_static
    CELERITAS_CUDA_OBJECT_LIBRARY ${target}_objects
  )

  set_target_properties(${target}_static PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    # CUDA_RESOLVE_DEVICE_SYMBOLS OFF # Default for static library
    CELERITAS_CUDA_LIBRARY_TYPE Static
    CELERITAS_CUDA_FINAL_LIBRARY ${target}_final
    CELERITAS_CUDA_MIDDLE_LIBRARY ${target}_cuda
    CELERITAS_CUDA_STATIC_LIBRARY ${target}_static
    CELERITAS_CUDA_OBJECT_LIBRARY ${target}_objects
  )

  set_target_properties(${target}_final PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON # Default for shared library
    CELERITAS_CUDA_LIBRARY_TYPE Final
    CELERITAS_CUDA_FINAL_LIBRARY ${target}_final
    CELERITAS_CUDA_STATIC_LIBRARY ${target}_static
    CELERITAS_CUDA_MIDDLE_LIBRARY ${target}_cuda
    CELERITAS_CUDA_OBJECT_LIBRARY ${target}_objects
  )

  target_link_libraries(${target}_final
    PUBLIC ${target}_cuda
  )

  target_link_options(${target}_final
    PRIVATE
    $<DEVICE_LINK:$<TARGET_FILE:${target}_static>>
  )
  if (CELERITAS_USE_VecGeom)
    target_link_libraries(${target}_objects
      PRIVATE VecGeom::vecgeom
    )
  endif()

  add_dependencies(${target}_final ${target}_cuda)
  add_dependencies(${target}_final ${target}_static)

  add_library(${target} ALIAS ${target}_final)

endfunction()

# Return TRUE if 'lib' depends/uses directly or indirectly the library `potentialdepend`
function(celeritas_depends_on OUTVARNAME lib potentialdepend)
  set(${OUTVARNAME} FALSE PARENT_SCOPE)
  if(TARGET ${lib} AND TARGET ${potentialdepend})
    get_target_property(lib_link_libraries ${lib} LINK_LIBRARIES)
    foreach(linklib ${lib_link_libraries})
      if(${linklib} STREQUAL ${potentialdepend})
        set(${OUTVARNAME} TRUE PARENT_SCOPE)
        return()
      endif()
      celeritas_depends_on(${OUTVARNAME} ${linklib} ${potentialdepend})
      if(${OUTVARNAME})
        set(${OUTVARNAME} ${${OUTVARNAME}} PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endif()
endfunction()



function(celeritas_strip_alias OUTVAR target)
  if(TARGET ${target})
    get_target_property(_target_alias ${target} ALIASED_TARGET)
  endif()
  if(TARGET ${_target_alias})
    set(target ${_target_alias})
  endif()
  set(${OUTVAR} ${target} PARENT_SCOPE)
endfunction()


function(celeritas_use_middle_lib_in_property target property)
  get_target_property(_target_libs ${target} ${property})

  set(_new_values)
  foreach(_lib ${_target_libs})
    set(_newlib ${_lib})
    if(TARGET ${_lib})
      celeritas_strip_alias(_lib ${_lib})
      get_target_property(_libmid ${_lib} CELERITAS_CUDA_MIDDLE_LIBRARY)
      if(_libmid)
        set(_newlib ${_libmid})
      endif()
    endif()
    list(APPEND _new_values ${_newlib})
  endforeach()

  if(_new_values)
    set_target_properties(${target} PROPERTIES
      ${property} "${_new_values}"
    )
  endif()
endfunction()

# Return the most derived "separatable cuda" library the target depends on.
# If two or more cuda library are independent, we return both and the calling executable
# should be linked with nvcc -dlink.
function(celeritas_find_final_library OUTLIST flat_dependency_list)
  set(_result "")
  foreach(_lib ${flat_dependency_list})
    if(NOT _result)
      list(APPEND _result ${_lib})
    else()
      set(_newresult "")
      foreach(_reslib ${_result})
        celeritas_depends_on(_depends_on ${_lib} ${_reslib})
        celeritas_depends_on(_depends_on ${_reslib} ${_lib})

        celeritas_depends_on(_depends_on ${_reslib} ${_lib})
        if(${_depends_on})
          # The library in the result depends/uses the library we are looking at,
          # let's keep the ones from result
          set(_newresult ${_result})
          break()
          # list(APPEND _newresult ${_reslib})
        else()
          celeritas_depends_on(_depends_on ${_lib} ${_reslib})
          if(${_depends_on})
            # We are in the opposite case, let's keep the library we are looking at
            list(APPEND _newresult ${_lib})
          else()
            # Unrelated keep both
            list(APPEND _newresult ${_reslib})
            list(APPEND _newresult ${_lib})
          endif()
        endif()
      endforeach()
      set(_result ${_newresult})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _result)
  set(_final_result "")
  foreach(_lib ${_result})
    if(TARGET ${_lib})
      get_target_property(_final_lib ${_lib} CELERITAS_CUDA_FINAL_LIBRARY)
      if(_final_lib)
        set(_lib ${_final_lib})
      endif()
    endif()
    list(APPEND _final_result ${_lib})
  endforeach()
  set(${OUTLIST} ${_final_result} PARENT_SCOPE)
endfunction()

# Replacement for target_link_libraries that is aware of
# the 3 libraries (static, middle, final) libraries needed
# for a separatable CUDA library
function(celeritas_target_link_libraries target)
  if(NOT BUILD_SHARED_LIBS OR NOT CELERITAS_USE_CUDA)
    target_link_libraries(${ARGV})
  else()
    celeritas_strip_alias(target ${target})

    celeritas_lib_contains_cuda(_contains_cuda ${target})

    set(_target_final ${target})
    set(_target_middle ${target})
    if (_contains_cuda)
      get_target_property(_targettype ${target} CELERITAS_CUDA_LIBRARY_TYPE)
      if(_targettype)
        get_target_property(_target_final ${target} CELERITAS_CUDA_FINAL_LIBRARY)
        get_target_property(_target_middle ${target} CELERITAS_CUDA_MIDDLE_LIBRARY)
        get_target_property(_target_object ${target} CELERITAS_CUDA_OBJECT_LIBRARY)
      endif()
    endif()

    # Set now to let taraget_link_libraries do the argument parsing
    target_link_libraries(${_target_middle} ${ARGN})
    if(_target_object)
      target_link_libraries(${_target_object} ${ARGN})
    endif()

    celeritas_use_middle_lib_in_property(${_target_middle} INTERFACE_LINK_LIBRARIES)
    celeritas_use_middle_lib_in_property(${_target_middle} LINK_LIBRARIES)

    celeritas_use_middle_lib_in_property(${_target_object} INTERFACE_LINK_LIBRARIES)
    celeritas_use_middle_lib_in_property(${_target_object} LINK_LIBRARIES)

    if(_contains_cuda)
      celeritas_cuda_gather_dependencies(_flat_target_link_libraries ${_target_middle})
      foreach(_lib ${_flat_target_link_libraries})
        get_target_property(_libstatic ${_lib} CELERITAS_CUDA_STATIC_LIBRARY)

        if(TARGET ${_libstatic})
          target_link_options(${_target_final}
            PRIVATE
            $<DEVICE_LINK:$<TARGET_FILE:${_libstatic}>>
          )
        endif()
      endforeach()
    endif()
  endif()

endfunction()

function(celeritas_cuda_gather_dependencies outlist target)
  if(TARGET ${target})
    celeritas_strip_alias(target ${target})
    get_target_property(_target_link_libraries ${target} LINK_LIBRARIES)
    if(_target_link_libraries)
      #message(WARNING "The link list for ${target} is ${_target_link_libraries}")
      foreach(_lib ${_target_link_libraries})
        celeritas_strip_alias(_lib ${_lib})
        get_target_property(_libmid ${_lib} CELERITAS_CUDA_MIDDLE_LIBRARY)
        if(TARGET ${_libmid})
          list(APPEND ${outlist} ${_libmid})
          # and recurse
          celeritas_cuda_gather_dependencies(_midlist ${_lib})
          #message(WARNING "The link mid list for ${target} for ${_lib} is ${_midlist}")
          list(APPEND ${outlist} ${_midlist})
        endif()
      endforeach()
    endif()
    list(REMOVE_DUPLICATES ${outlist})
    set(${outlist} ${${outlist}} PARENT_SCOPE)
  endif()
endfunction()


#-----------------------------------------------------------------------------#
