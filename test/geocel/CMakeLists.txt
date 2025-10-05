#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_Geant4)
  set(_needs_geant4 DISABLE)
else()
  celeritas_get_g4libs(_g4_global_libs global)
endif()

if(CELERITAS_UNITS STREQUAL "CGS")
  set(_fixme_cgs)
else()
  # Unit tests haven't yet been updated to include the correct units
  set(_fixme_cgs DISABLE)
endif()

if(NOT CELERITAS_USE_PNG)
  set(_needs_png DISABLE)
endif()

#-----------------------------------------------------------------------------#
# LIBRARY
#-----------------------------------------------------------------------------#
set(SOURCES
  LazyGeantGeoManager.cc
  GeantImportVolumeResult.cc
  GenericGeoResults.cc
  GenericGeoTestInterface.cc
  GeoTests.cc
  SurfaceTestBase.cc
  VolumeTestBase.cc
)

set(PRIVATE_DEPS ${testcel_core} Celeritas::ExtGeant4Geo)
if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  list(APPEND SOURCES
    g4/GeantGeoTestBase.cc
  )
endif()
if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    vg/VecgeomTestBase.cc
  )
  list(APPEND PRIVATE_DEPS Celeritas::ExtVecGeom)
endif()

celeritas_add_test_library(testcel_geocel ${SOURCES})

celeritas_target_link_libraries(testcel_geocel
  PUBLIC testcel_harness Celeritas::geocel
  PRIVATE ${PRIVATE_DEPS}
)

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_geocel testcel_core Celeritas::geocel Celeritas::corecel
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#

celeritas_add_test(BoundingBox.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json
)
celeritas_add_test(Surface.test.cc)
celeritas_add_test(Volume.test.cc)

celeritas_add_test(random/IsotropicDistribution.test.cc)
celeritas_add_test(random/UniformBoxDistribution.test.cc)

#-----------------------------------------------------------------------------#
# VecGeom

if(CELERITAS_USE_VecGeom)
  # Vecgeom uses global counters that interfere with copying data to GPU and
  # change the internal numbering. Run all tests separately using a filter,
  # first constructing a list of tests that *could* be run.
  set(_vgdml_tests
    "FourLevelsVgdml"
    "MultiLevelVgdml"
    "SimpleCmsVgdml"
    "SolidsVgdml"
    "TwoBoxesVgdml"
  )
  set(_g4vg_tests
    "CmsEeBackDee"
    "Cmse"
    "FourLevels"
    "MultiLevel"
    "Polyhedra"
    "Replica"
    "Solids"
    "TransformedBox"
    "Znenv"
  )
  set(_vecgeom_test_filter FILTER)
  foreach(_test IN LISTS _vgdml_tests _g4vg_tests)
    list(APPEND _vecgeom_test_filter "${_test}Test.*")
  endforeach()

  # Now build a list of tests that *can't* successfully run
  set(_vecgeom_disable)
  if(NOT VecGeom_GDML_FOUND)
    list(APPEND _vecgeom_disable
      ".*Vgdml"
    )
  elseif(CELERITAS_VecGeom_SURFACE)
    list(APPEND _vecgeom_disable
      MultiLevelVgdml # Scaling used to create mirror symmetry
      SolidsVgdml # Missing shapes
    )
  endif()

  if(NOT CELERITAS_USE_Geant4)
    list(APPEND _vecgeom_disable ${_g4vg_tests})
  elseif(CELERITAS_VecGeom_SURFACE)
    list(APPEND _vecgeom_disable
      Solids # Missing shapes
    )
  endif()

  if(_vecgeom_disable)
    # Turn list of tests into (A|B|C)
    list(JOIN _vecgeom_disable "|" _joined_bits)
    list(APPEND _vecgeom_test_filter
      DISABLE_FILTER_REGEX "^(${_joined_bits})Test.*"
    )
  endif()

  celeritas_add_device_test(vg/Vecgeom
    LINK_LIBRARIES
      VecGeom::vecgeom Celeritas::ExtGeant4Geo Celeritas::ExtThrust
      nlohmann_json::nlohmann_json
    ${_vecgeom_test_filter}
    ${_fixme_cgs}
  )
endif()

#-----------------------------------------------------------------------------#
# Geant4

list(APPEND CELERITASTEST_LINK_LIBRARIES Celeritas::ExtGeant4Geo)
celeritas_add_test(GeantGdmlLoader.test.cc ${_needs_geant4})

if(CELERITAS_USE_Geant4)
  celeritas_add_test(ScopedGeantLogger.test.cc LINK_LIBRARIES ${_g4_global_libs})
endif()

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  # These all require the Geant4 geometry wrappers which only are built when
  # double-precision is enabled
  celeritas_add_test(g4/GeantGeo.test.cc)
  celeritas_add_test(g4/GeantNavHistoryUpdater.test.cc)
  celeritas_add_test(GeantGeoUtils.test.cc)
endif()
list(POP_BACK CELERITASTEST_LINK_LIBRARIES)

#-----------------------------------------------------------------------------#
# Rasterization

celeritas_add_test(rasterize/Color.test.cc)
celeritas_add_test(rasterize/Image.test.cc)
celeritas_add_test(rasterize/Raytracer.test.cc)

# In that very specific case (which is the one used in Athena integration)
# this test fails to link as geocel seems to be removed by the linker
# https://github.com/celeritas-project/celeritas/pull/1498
if(NOT (CELERITAS_USE_VecGeom AND CELERITAS_USE_CUDA
   AND NOT CELERITAS_USE_PNG AND NOT BUILD_SHARED_LIBS))
  celeritas_add_test(rasterize/ImageWriter.test.cc ${_needs_png})
endif()

#-----------------------------------------------------------------------------#
