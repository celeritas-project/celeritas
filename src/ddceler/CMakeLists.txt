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
  set(DD4HEP_ROOT_LIBRARY_PATH "$ENV{ROOT_LIBRARY_PATH}"
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
  SET( install_destination "lib" )
  if( CMAKE_INSTALL_LIBDIR )
    SET( install_destination ${CMAKE_INSTALL_LIBDIR} )
  endif()
  install(FILES $<TARGET_FILE_DIR:${library}>/${rootmapfile}
    DESTINATION ${install_destination}
  )
endfunction()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

dd4hep_add_plugin(DDcelerTMI
  SOURCES DDcelerTMI.cc
  USES DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

dd4hep_add_plugin(DDcelerRunAction
  SOURCES DDcelerRunAction.cc
  USES DD4hep::DDG4 DD4hep::DDCore Celeritas::accel Celeritas::celeritas Celeritas::corecel
)

# Configure all plugins
set(_plugins DDcelerTMI DDcelerRunAction)
set(_component_sources)
set(_component_outputs)

foreach(_target ${_plugins})
  # VecGeom linking if enabled
  if(CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA)
    target_link_libraries(${_target} PRIVATE
      "-Wl,--push-state,--no-as-needed" Celeritas::accel_final "-Wl,--pop-state"
    )
  endif()
  # Track .components files for copying
  list(APPEND _component_sources ${CMAKE_CURRENT_BINARY_DIR}/lib${_target}.components)
  list(APPEND _component_outputs ${CMAKE_BINARY_DIR}/lib/lib${_target}.components)
endforeach()

# Ensure plugins link against build directory libraries, not installed ones
set_target_properties(${_plugins} PROPERTIES
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH FALSE
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(${_plugins} PROPERTIES APPEND
    LINK_FLAGS "-Wl,--disable-new-dtags"
  )
endif()

install(TARGETS ${_plugins} LIBRARY DESTINATION lib)

# Copy .components files to lib directory
add_custom_command(
  OUTPUT ${_component_outputs}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_component_sources} ${CMAKE_BINARY_DIR}/lib/
  DEPENDS ${_plugins} ${_component_sources}
  COMMENT "Copying .components files to build/lib"
)

add_custom_target(copy_components ALL DEPENDS ${_component_outputs})
