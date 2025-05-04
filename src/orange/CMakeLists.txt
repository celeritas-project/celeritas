#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS Celeritas::geocel nlohmann_json::nlohmann_json)
set(PUBLIC_DEPS Celeritas::corecel)

#-----------------------------------------------------------------------------#
# Main code
#-----------------------------------------------------------------------------#

list(APPEND SOURCES
  BoundingBoxUtils.cc
  MatrixUtils.cc
  OrangeInputIO.json.cc
  OrangeParams.cc
  OrangeParamsOutput.cc
  OrangeTypes.cc
  SafetyImager.cc
  detail/BIHBuilder.cc
  detail/BIHPartitioner.cc
  detail/DepthCalculator.cc
  detail/LogicUtils.cc
  detail/OrangeInputIOImpl.json.cc
  detail/RectArrayInserter.cc
  detail/SurfacesRecordBuilder.cc
  detail/UnitInserter.cc
  detail/UniverseInserter.cc
  orangeinp/CsgObject.cc
  orangeinp/CsgTree.cc
  orangeinp/CsgTreeIO.json.cc
  orangeinp/CsgTreeUtils.cc
  orangeinp/CsgTypes.cc
  orangeinp/InputBuilder.cc
  orangeinp/IntersectRegion.cc
  orangeinp/IntersectSurfaceBuilder.cc
  orangeinp/ObjectInterface.cc
  orangeinp/ObjectIO.json.cc
  orangeinp/PolySolid.cc
  orangeinp/ProtoInterface.cc
  orangeinp/Shape.cc
  orangeinp/Solid.cc
  orangeinp/ScaleUtils.cc
  orangeinp/Transformed.cc
  orangeinp/UnitProto.cc
  orangeinp/detail/BoundingZone.cc
  orangeinp/detail/BuildIntersectRegion.cc
  orangeinp/detail/CsgUnitBuilder.cc
  orangeinp/detail/DeMorganSimplifier.cc
  orangeinp/detail/InternalSurfaceFlagger.cc
  orangeinp/detail/IntersectSurfaceState.cc
  orangeinp/detail/LocalSurfaceInserter.cc
  orangeinp/detail/NodeSimplifier.cc
  orangeinp/detail/ProtoBuilder.cc
  orangeinp/detail/ProtoMap.cc
  orangeinp/detail/SenseEvaluator.cc
  orangeinp/detail/SurfaceGridHash.cc
  orangeinp/detail/TransformInserter.cc
  orangeinp/detail/VolumeBuilder.cc
  surf/ConeAligned.cc
  surf/CylAligned.cc
  surf/FaceNamer.cc
  surf/GeneralQuadric.cc
  surf/Involute.cc
  surf/Plane.cc
  surf/PlaneAligned.cc
  surf/SimpleQuadric.cc
  surf/SoftSurfaceEqual.cc
  surf/Sphere.cc
  surf/SurfaceClipper.cc
  surf/SurfaceIO.cc
  surf/SurfaceSimplifier.cc
  surf/VariantSurface.cc
  surf/detail/SurfaceTransformer.cc
  surf/detail/SurfaceTranslator.cc
  transform/SignedPermutation.cc
  transform/TransformHasher.cc
  transform/TransformIO.cc
  transform/TransformSimplifier.cc
  transform/Transformation.cc
  transform/VariantTransform.cc
)


if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set(_cg4org_sources
    g4org/Converter.cc
    g4org/LogicalVolumeConverter.cc
    g4org/PhysicalVolumeConverter.cc
    g4org/ProtoConstructor.cc
    g4org/SolidConverter.cc
  )
  celeritas_get_g4libs(_cg4org_libs geometry materials)
  list(APPEND _cg4org_libs Celeritas::corecel Celeritas::geocel)

  celeritas_add_object_library(geocel_g4org ${_cg4org_sources})
  celeritas_target_link_libraries(geocel_g4org PRIVATE ${_cg4org_libs})

  list(APPEND SOURCES $<TARGET_OBJECTS:geocel_g4org>)
  list(APPEND PRIVATE_DEPS geocel_g4org)
endif()

celeritas_polysource_append(SOURCES RaytraceImager)

#-----------------------------------------------------------------------------#
# Create library
#-----------------------------------------------------------------------------#

celeritas_add_src_library(orange ${SOURCES})
celeritas_target_link_libraries(orange
  PRIVATE ${PRIVATE_DEPS}
  PUBLIC ${PUBLIC_DEPS}
)

#-----------------------------------------------------------------------------#
