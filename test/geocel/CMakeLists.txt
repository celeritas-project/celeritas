#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
else()
  celeritas_get_g4libs(_g4_geo_libs geometry)
  celeritas_get_g4libs(_g4_global_libs global)
endif()

if(CELERITAS_UNITS STREQUAL "CGS")
  set(_fixme_cgs)
else()
  # Unit tests haven't yet been updated to include the correct units
  set(_fixme_cgs DISABLE)
endif()

#-----------------------------------------------------------------------------#
# LIBRARY
#-----------------------------------------------------------------------------#
set(SOURCES
  LazyGeoManager.cc
  GenericGeoTestBase.cc
)

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  list(APPEND SOURCES
    g4/GeantGeoTestBase.cc
  )
endif()
if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    vg/VecgeomTestBase.cc
  )
  list(APPEND PRIVATE_DEPS VecGeom::vecgeom)
endif()

celeritas_add_test_library(testcel_geocel ${SOURCES})

celeritas_target_link_libraries(testcel_geocel
  PUBLIC
    testcel_harness Celeritas::geocel
  PRIVATE
    testcel_core ${_g4_geo_libs} ${VecGeom_LIBRARIES}
)

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_geocel testcel_core Celeritas::geocel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(BoundingBox.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json
)

#-----------------------------------------------------------------------------#
# VecGeom

if(CELERITAS_USE_VecGeom)
  # Vecgeom uses global counters that interfere with copying data to GPU and
  # change the internal numbering.
  set(_vecgeom_tests)
  if(VecGeom_GDML_FOUND)
    list(APPEND _vecgeom_tests
      "SimpleCmsTest.*"
    )
    if(NOT VecGeom_SURF_FOUND)
      # TODO: vecgeom surface doesn't support some of these shapes
      list(APPEND _vecgeom_tests
        "FourLevelsTest.*"
        "SolidsTest.*"
        "CmseTest.*"
      )
    endif()
  endif()
  if(CELERITAS_USE_Geant4)
    list(APPEND _vecgeom_tests
      "FourLevelsGeantTest.*"
      "SolidsGeantTest.*"
      "ZnenvGeantTest.*"
    )
  endif()
  if(NOT _vecgeom_tests)
    # Bad: no way of setting up geometry!
    set(_vecgeom_tests DISABLE)
  endif()
  celeritas_add_device_test(vg/Vecgeom
    LINK_LIBRARIES VecGeom::vecgeom
    FILTER ${_vecgeom_tests}
    ${_fixme_cgs}
  )
endif()

#-----------------------------------------------------------------------------#
# Geant4

if(CELERITAS_USE_Geant4)
  celeritas_add_test(ScopedGeantLogger.test.cc LINK_LIBRARIES ${_g4_global_libs})
endif()

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  celeritas_add_test(g4/GeantGeo.test.cc LINK_LIBRARIES ${_g4_geo_libs})
  celeritas_add_test(GeantGeoUtils.test.cc LINK_LIBRARIES ${_g4_geo_libs})
endif()

#-----------------------------------------------------------------------------#
# Rasterization


celeritas_add_test(rasterize/Image.test.cc)
celeritas_add_test(rasterize/Raytracer.test.cc)

#-----------------------------------------------------------------------------#
