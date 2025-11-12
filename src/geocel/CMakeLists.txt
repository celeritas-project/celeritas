#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS nlohmann_json::nlohmann_json)
set(PUBLIC_DEPS Celeritas::corecel)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  BoundingBoxIO.json.cc
  GeoInterface.cc
  GeoParamsOutput.cc
  SurfaceParams.cc
  VolumeParams.cc
  VolumeIdBuilder.cc
  VolumeToString.cc
  detail/SurfaceParamsBuilder.cc
  rasterize/Color.cc
  rasterize/Image.cc
  rasterize/ImageIO.json.cc
)

#-----------------------------------------------------------------------------#
# Configuration-dependent code
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_PNG)
  celeritas_add_object_library(geocel_png
    rasterize/ImageWriter.libpng.cc
  )
  celeritas_target_link_libraries(geocel_png
    PRIVATE PNG::PNG Celeritas::BuildFlags
  )
  list(APPEND SOURCES $<TARGET_OBJECTS:geocel_png>)
  list(APPEND PRIVATE_DEPS geocel_png)
endif()

if(CELERITAS_USE_VecGeom AND CELERITAS_USE_Geant4)
  list(APPEND PRIVATE_DEPS G4VG::g4vg)
endif()

if(CELERITAS_USE_Geant4)
  celeritas_add_object_library(geocel_geant4
    GeantUtils.cc
    GeantGeoUtils.cc
    GeantGdmlLoader.cc
    ScopedGeantExceptionHandler.cc
    ScopedGeantLogger.cc
    GeantGeoParams.cc
    detail/GeantVolumeInstanceMapper.cc
    g4/GeantNavHistoryUpdater.cc
    g4/RaytraceImager.cc
    g4/SafetyImager.cc
    g4/detail/GeantGeoNavCollection.cc
  )

  celeritas_get_g4libs(_cg4_libs global geometry digits_hits materials particles
    persistency intercoms)
  celeritas_target_link_libraries(geocel_geant4
    PRIVATE
      ${_cg4_libs} XercesC::XercesC nlohmann_json::nlohmann_json
      Celeritas::BuildFlags
  )

  list(APPEND SOURCES $<TARGET_OBJECTS:geocel_geant4>)
  list(APPEND PRIVATE_DEPS geocel_geant4)
endif()

if(CELERITAS_USE_VecGeom)
  list(APPEND SOURCES
    vg/VecgeomParams.cc
    vg/VecgeomParamsOutput.cc
    vg/detail/VecgeomNavCollection.cc
    vg/RaytraceImager.cc
    vg/SafetyImager.cc
  )
  if(VecGeom_GDML_FOUND)
    list(APPEND PRIVATE_DEPS VecGeom::vgdml)
  endif()
  if(CELERITAS_USE_CUDA)
    list(APPEND SOURCES
      vg/RaytraceImager.cu
      vg/detail/VecgeomSetup.cu
    )
    # VecGeom CUDA is required in this case; those libraries are automatically
    # configured via the CudaRdc linking
  elseif(VecGeom_CUDA_FOUND)
    # VecGeom is built with CUDA but Celeritas is not: need to fill missing
    # symbols with the VecGeom "final" library
    list(APPEND PUBLIC_DEPS VecGeom::vecgeomcuda)
  endif()
  list(APPEND PRIVATE_DEPS VecGeom::vecgeom)
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(geocel ${SOURCES})

celeritas_target_link_libraries(geocel
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
