dd4hep_add_plugin(DDcelerTMI
    SOURCES DDcelerTMI.cc
    USES    DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )
dd4hep_add_plugin(DDcelerRunAction
    SOURCES DDcelerRunAction.cc
    USES    DD4hep::DDG4 DD4hep::DDCore Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )
install(TARGETS DDcelerTMI DDcelerRunAction LIBRARY DESTINATION lib)
