#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#----------------------------------------------------------------------------#

# Set variables used by internal dd4hep macros
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}")
set(LIBRARY_OUTPUT_PATH "${CELERITAS_LIBRARY_OUTPUT_DIRECTORY}")

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
  if(CMAKE_INSTALL_LIBDIR)
    set(install_destination ${CMAKE_INSTALL_LIBDIR})
  endif()
  install(FILES $<TARGET_FILE_DIR:${library}>/${rootmapfile}
    DESTINATION ${install_destination}
  )
endfunction()

# <<< END OVERRIDE
#----------------------------------------------------------------------------#

get_target_property(_final_lib Celeritas::accel CUDA_RDC_FINAL_LIBRARY)
if(_final_lib)
  # Using CUDA RDC: check for other flags as well
  if(NOT DEFINED CELERITAS_DISABLE_NEW_DTAGS)
    set(_dtag_flag "--disable-new-dtags")
    include(CheckLinkerFlag)
    check_linker_flag(CXX "LINKER:${_dtag_flag}" CELERITAS_DISABLE_NEW_DTAGS)
    if(CELERITAS_DISABLE_NEW_DTAGS)
      set(CELERITAS_DISABLE_NEW_DTAGS "${_dtag_flag}" CACHE INTERNAL
        "flag used to disable dtags")
    endif()
  endif()
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  cmake_language(GET_MESSAGE_LOG_LEVEL _orig_log_lev)
else()
  set(_orig_log_lev STATUS)
endif()

function(celer_dd4hep_plugin target)
  # Suppress verbose dd4hep
  if(_orig_log_lev STREQUAL STATUS)
    set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
  endif()

  dd4hep_add_plugin(${target} ${ARGN})

  if(_final_lib)
    # Add dependency on the final library to ensure it's built first
    add_dependencies(${target} ${_final_lib})

    # Link CUDA RDC "final" library into plugin
    target_link_options(${target} PRIVATE
      "-Wl,--push-state,--no-as-needed"
      "$<TARGET_LINKER_FILE:${_final_lib}>"
      "-Wl,--pop-state"
    )
    if(CELERITAS_DISABLE_NEW_DTAGS)
      set_property(TARGET ${_plugins} APPEND
        PROPERTY LINK_FLAGS "-Wl,--disable-new-dtags"
      )
    endif()
  endif()
endfunction()

#----------------------------------------------------------------------------#
# Add plugins

celer_dd4hep_plugin(CelerPhysics
  SOURCES CelerPhysics.cc
  USES DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

celer_dd4hep_plugin(CelerRun
  SOURCES CelerRun.cc
  USES DD4hep::DDG4 DD4hep::DDCore
    Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

#----------------------------------------------------------------------------#
