
dd4hep_add_plugin(DDG4Celeritas
    SOURCES *.cc
    USES    DD4hep::DDG4 Celeritas::accel Celeritas::celeritas Celeritas::corecel
    )
install(TARGETS DDG4Celeritas LIBRARY DESTINATION lib)
