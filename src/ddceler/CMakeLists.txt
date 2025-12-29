#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

include("${DD4hep_DIR}/cmake/DD4hep.cmake")

#----------------------------------------------------------------------------#
# >>> OVERRIDE DD4HEP TO FIX NEW/RPATH/MACOS ROOT
# see https://github.com/AIDASoft/DD4hep/pull/1545
#----------------------------------------------------------------------------#

function(dd4hep_generate_rootmap library)
  if(APPLE)
    set(_ld_var DYLD_LIBRARY_PATH)
  else()
    set(_ld_var LD_LIBRARY_PATH)
  endif()

  set(DD4HEP_LIBRARY_PATH "$ENV{${_ld_var}}"
    CACHE STRING
    "Set ${_ld_var} when adding plugins"
  )

  if("$ENV{ROOT_LIBRARY_PATH}")
    set(_default_root_lib_path "$ENV{ROOT_LIBRARY_PATH}")
  else()
    set(_default_root_lib_path "${ROOT_LIBRARY_DIR}")
  endif()
  set(DD4HEP_ROOT_LIBRARY_PATH "${_default_root_lib_path}"
    CACHE STRING
    "Set ROOT_LIBRARY_PATH when adding plugins"
  )

  string(JOIN ":" _ld_path
    "$<TARGET_FILE_DIR:${library}>"
    "$<TARGET_FILE_DIR:DD4hep::DD4hepGaudiPluginMgr>"
    "${DD4HEP_LIBRARY_PATH}"
  )

  set(rootmapfile ${CMAKE_SHARED_MODULE_PREFIX}${library}.components)

  add_custom_command(OUTPUT ${rootmapfile}
    DEPENDS ${library}
    COMMAND
    "${CMAKE_COMMAND}" -E env
    "${_ld_var}=${_ld_path}"
    "ROOT_LIBRARY_PATH=${DD4HEP_ROOT_LIBRARY_PATH}"
    $<TARGET_FILE:DD4hep::listcomponents> -o ${rootmapfile} $<TARGET_FILE_NAME:${library}>
    WORKING_DIRECTORY ${LIBRARY_OUTPUT_PATH}
  )

  add_custom_target(Components_${library} ALL DEPENDS ${rootmapfile})
  set(install_destination "lib")
  if(CMAKE_INSTALL_LIBDIR )
    set(install_destination ${CMAKE_INSTALL_LIBDIR})
  endif()
  install(FILES $<TARGET_FILE_DIR:${library}>/${rootmapfile}
    DESTINATION ${install_destination}
  )
endfunction()

# <<< END OVERRIDE
#----------------------------------------------------------------------------#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

dd4hep_add_plugin(DDcelerTMI
  SOURCES DDcelerTMI.cc
  USES DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

dd4hep_add_plugin(DDcelerRunAction
  SOURCES DDcelerRunAction.cc
  USES DD4hep::DDG4 DD4hep::DDCore
    Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
  foreach(_target DDcelerTMI DDcelerRunAction)
    target_link_options(${_target} PRIVATE
      "-Wl,--push-state,--no-as-needed"
      "-l$<TARGET:Celeritas::accel_final>"
      "-Wl,--pop-state"
    )
  endforeach()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(${_plugins} PROPERTIES APPEND
    LINK_FLAGS "-Wl,--disable-new-dtags"
  )
endif()

#----------------------------------------------------------------------------#
