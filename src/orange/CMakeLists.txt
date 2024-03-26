#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::geocel)
set(PUBLIC_DEPS Celeritas::corecel)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  BoundingBoxUtils.cc
  MatrixUtils.cc
  OrangeParams.cc
  OrangeParamsOutput.cc
  OrangeTypes.cc
  detail/BIHBuilder.cc
  detail/BIHPartitioner.cc
  detail/DepthCalculator.cc
  detail/RectArrayInserter.cc
  detail/SurfacesRecordBuilder.cc
  detail/UnitInserter.cc
  detail/UniverseInserter.cc
  orangeinp/ConvexRegion.cc
  orangeinp/ConvexSurfaceBuilder.cc
  orangeinp/CsgObject.cc
  orangeinp/CsgTree.cc
  orangeinp/CsgTypes.cc
  orangeinp/CsgTreeUtils.cc
  orangeinp/ProtoInterface.cc
  orangeinp/Shape.cc
  orangeinp/Solid.cc
  orangeinp/Transformed.cc
  orangeinp/UnitProto.cc
  orangeinp/detail/BoundingZone.cc
  orangeinp/detail/BuildConvexRegion.cc
  orangeinp/detail/ConvexSurfaceState.cc
  orangeinp/detail/CsgUnitBuilder.cc
  orangeinp/detail/InputBuilder.cc
  orangeinp/detail/InternalSurfaceFlagger.cc
  orangeinp/detail/LocalSurfaceInserter.cc
  orangeinp/detail/NodeSimplifier.cc
  orangeinp/detail/PostfixLogicBuilder.cc
  orangeinp/detail/ProtoMap.cc
  orangeinp/detail/SenseEvaluator.cc
  orangeinp/detail/TransformInserter.cc
  orangeinp/detail/VolumeBuilder.cc
  surf/ConeAligned.cc
  surf/CylAligned.cc
  surf/FaceNamer.cc
  surf/GeneralQuadric.cc
  surf/Plane.cc
  surf/PlaneAligned.cc
  surf/SimpleQuadric.cc
  surf/Sphere.cc
  surf/SoftSurfaceEqual.cc
  surf/SurfaceClipper.cc
  surf/SurfaceIO.cc
  surf/SurfaceSimplifier.cc
  surf/VariantSurface.cc
  surf/detail/SurfaceTranslator.cc
  surf/detail/SurfaceTransformer.cc
  transform/SignedPermutation.cc
  transform/TransformHasher.cc
  transform/TransformIO.cc
  transform/TransformSimplifier.cc
  transform/Transformation.cc
  transform/VariantTransform.cc
)

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    OrangeInputIO.json.cc
    detail/OrangeInputIOImpl.json.cc
    orangeinp/CsgTreeIO.json.cc
    orangeinp/ObjectIO.json.cc
  )
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
endif()

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set( _cg4org_sources
    g4org/SolidConverter.cc
    g4org/Transformer.cc
  )
  celeritas_get_g4libs(_cg4org_libs geometry)
  list(APPEND _cg4org_libs Celeritas::corecel Celeritas::geocel)

  celeritas_add_object_library(geocel_g4org ${_cg4org_sources})
  celeritas_target_link_libraries(geocel_g4org PRIVATE ${_cg4org_libs})

  list(APPEND SOURCES $<TARGET_OBJECTS:geocel_g4org>)
  list(APPEND PRIVATE_DEPS geocel_g4org)
endif()

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_library(orange ${SOURCES})
celeritas_target_link_libraries(orange
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
