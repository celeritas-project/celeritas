
if(PERFETTO_ROOT AND NOT Perfetto_ROOT)
    set(Perfetto_ROOT ${PERFETTO_ROOT} CACHE PATH "Perfetto.[cc|hh] directory")
endif()

find_path(Perfetto_INCLUDE_DIR perfetto.h
            HINTS "${Perfetto_ROOT}")

if(Perfetto_INCLUDE_DIR)
    add_library(celeritas_perfetto STATIC ${Perfetto_INCLUDE_DIR}/perfetto.cc)
    add_library(Celeritas::Perfetto ALIAS celeritas_perfetto)
    target_include_directories(celeritas_perfetto INTERFACE ${Perfetto_INCLUDE_DIR})
endif()