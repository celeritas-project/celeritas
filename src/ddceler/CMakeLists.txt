
dd4hep_add_plugin(DDcelerTMI
    SOURCES *.cc
    USES    DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )
install(TARGETS DDcelerTMI LIBRARY DESTINATION lib)
