#------------------------------- -*- cmake -*- -------------------------------#
# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_test_library(testcel_orange
  OrangeGeoTestBase.cc
  OrangeTestBase.cc
  g4org/GeantLoadTestBase.cc
  orangeinp/CsgTestUtils.cc
  orangeinp/ObjectTestBase.cc
)
celeritas_target_link_libraries(testcel_orange
  PUBLIC
    Celeritas::orange testcel_harness testcel_core testcel_geocel
  PRIVATE
    Celeritas::orange nlohmann_json::nlohmann_json
)

#-----------------------------------------------------------------------------#
# VARIABLES
#-----------------------------------------------------------------------------#

if(CELERITAS_USE_Geant4 AND CELERITAS_REAL_TYPE STREQUAL "double")
  set(_use_g4org TRUE)
  set(_needs_g4org)
else()
  set(_use_g4org FALSE)
  set(_needs_g4org DISABLE)
endif()

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL
  LINK_LIBRARIES testcel_orange Celeritas::orange
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#
# Base
celeritas_add_test(BoundingBoxUtils.test.cc)
celeritas_add_test(MatrixUtils.test.cc)
celeritas_add_test(OrangeTypes.test.cc)
celeritas_add_test(RaytraceImager.test.cc GPU)

# High level
celeritas_add_test(Orange.test.cc)
celeritas_add_test(OrangeGeant.test.cc ${_needs_g4org}
  LINK_LIBRARIES nlohmann_json::nlohmann_json)
celeritas_add_test(OrangeJson.test.cc)
celeritas_add_device_test(OrangeShift)

celeritas_add_test(detail/UniverseIndexer.test.cc)

# Bounding interval hierarchy
celeritas_add_test(detail/BIHBuilder.test.cc)
celeritas_add_test(detail/BIHEnclosingVolFinder.test.cc)
celeritas_add_test(detail/BIHIntersectingVolFinder.test.cc)
celeritas_add_test(detail/BIHUtils.test.cc)

# Oriented Bounding Zone
celeritas_add_test(detail/OrientedBoundingZone.test.cc)

# Logic utilities
celeritas_add_test(detail/LogicUtils.test.cc)

#-----------------------------------------------------------------------------#
# Input construction
celeritas_add_test(orangeinp/CsgObject.test.cc)
celeritas_add_test(orangeinp/CsgTree.test.cc)
celeritas_add_test(orangeinp/CsgTreeUtils.test.cc)
celeritas_add_test(orangeinp/IntersectRegion.test.cc)
celeritas_add_test(orangeinp/IntersectSurfaceBuilder.test.cc)
celeritas_add_test(orangeinp/PolySolid.test.cc)
celeritas_add_test(orangeinp/Shape.test.cc)
celeritas_add_test(orangeinp/Solid.test.cc)
celeritas_add_test(orangeinp/Transformed.test.cc)
celeritas_add_test(orangeinp/UnitProto.test.cc
  LINK_LIBRARIES nlohmann_json::nlohmann_json)

celeritas_add_test(orangeinp/detail/BoundingZone.test.cc)
celeritas_add_test(orangeinp/detail/ConvexHullFinder.test.cc)
celeritas_add_test(orangeinp/detail/PolygonUtils.test.cc)
celeritas_add_test(orangeinp/detail/ProtoMap.test.cc)
celeritas_add_test(orangeinp/detail/LocalSurfaceInserter.test.cc)
celeritas_add_test(orangeinp/detail/SurfaceGridHash.test.cc)
celeritas_add_test(orangeinp/detail/TransformInserter.test.cc)

#-----------------------------------------------------------------------------#
# Transforms
celeritas_add_test(transform/SignedPermutation.test.cc)
celeritas_add_test(transform/TransformSimplifier.test.cc)
celeritas_add_test(transform/Transformation.test.cc)
celeritas_add_test(transform/Translation.test.cc)
celeritas_add_test(transform/VariantTransform.test.cc)

#-----------------------------------------------------------------------------#
# Surfaces
celeritas_add_test(surf/ConeAligned.test.cc)
celeritas_add_test(surf/CylAligned.test.cc)
celeritas_add_test(surf/CylCentered.test.cc)
celeritas_add_test(surf/GeneralQuadric.test.cc)
celeritas_add_test(surf/Plane.test.cc)
celeritas_add_test(surf/PlaneAligned.test.cc)
celeritas_add_test(surf/SimpleQuadric.test.cc)
celeritas_add_test(surf/Sphere.test.cc)
celeritas_add_test(surf/SphereCentered.test.cc)
celeritas_add_test(surf/Involute.test.cc)

# Surface utilities
celeritas_add_test(surf/FaceNamer.test.cc)
celeritas_add_test(surf/RecursiveSimplifier.test.cc)
celeritas_add_test(surf/SoftSurfaceEqual.test.cc)
celeritas_add_test(surf/SurfaceClipper.test.cc)
celeritas_add_test(surf/SurfaceSimplifier.test.cc)
celeritas_add_device_test(surf/LocalSurfaceVisitor)

# Surface details
celeritas_add_test(surf/detail/QuadraticSolver.test.cc)
celeritas_add_test(surf/detail/SurfaceTranslator.test.cc)
celeritas_add_test(surf/detail/SurfaceTransformer.test.cc)
celeritas_add_test(surf/detail/InvoluteSolver.test.cc)
celeritas_add_test(surf/detail/InvolutePoint.test.cc)

#-----------------------------------------------------------------------------#
# Universe
celeritas_add_test(univ/VolumeView.test.cc)
celeritas_add_test(univ/RectArrayTracker.test.cc)
celeritas_add_device_test(univ/SimpleUnitTracker)
celeritas_add_test(univ/TrackerVisitor.test.cc)

# Universe details
celeritas_add_test(univ/detail/InfixEvaluator.test.cc)
celeritas_add_test(univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(univ/detail/LogicStack.test.cc)
celeritas_add_test(univ/detail/RaggedRightIndexer.test.cc)
celeritas_add_test(univ/detail/SurfaceFunctors.test.cc)
celeritas_add_test(univ/detail/SenseCalculator.test.cc)

#-----------------------------------------------------------------------------#
# Geant4 construction
if(_use_g4org)
  celeritas_add_test(g4org/Converter.test.cc)
  celeritas_add_test(g4org/SolidConverter.test.cc
    LINK_LIBRARIES Celeritas::ExtGeant4Geo)
  celeritas_add_test(g4org/PhysicalVolumeConverter.test.cc)
  celeritas_add_test(g4org/ProtoConstructor.test.cc)
  celeritas_add_test(g4org/Transformer.test.cc
    LINK_LIBRARIES Celeritas::ExtGeant4Geo)
endif()

#-----------------------------------------------------------------------------#
