set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

dd4hep_add_plugin(DDcelerTMI
    SOURCES DDcelerTMI.cc
    USES    DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )
dd4hep_add_plugin(DDcelerRunAction
    SOURCES DDcelerRunAction.cc
    USES    DD4hep::DDG4 DD4hep::DDCore Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )

# When VecGeom is enabled, add accel_final with --no-as-needed
if(CELERITAS_USE_VecGeom)
    foreach(_target DDcelerTMI DDcelerRunAction)
        target_link_libraries(${_target} PRIVATE
            "-Wl,--push-state,--no-as-needed" Celeritas::accel_final "-Wl,--pop-state"
        )
    endforeach()
endif()

# Ensure plugins link against build directory libraries, not installed ones
# Must come after dd4hep_add_plugin which sets its own RPATH
# Use RPATH instead of RUNPATH for higher priority in library search
set_target_properties(DDcelerTMI DDcelerRunAction PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

install(TARGETS DDcelerTMI DDcelerRunAction LIBRARY DESTINATION lib)


add_custom_target(copy_components ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/libDDcelerTMI.components
        ${CMAKE_CURRENT_BINARY_DIR}/libDDcelerRunAction.components
        ${CMAKE_BINARY_DIR}/lib/
    DEPENDS DDcelerTMI DDcelerRunAction
    COMMENT "Copying .components files to build/lib"
)
