#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_orange
  OrangeGeoTestBase.cc
  OrangeTestBase.cc
  orangeinp/CsgTestUtils.cc
  orangeinp/ObjectTestBase.cc
)
celeritas_target_link_libraries(testcel_orange
  PUBLIC
    Celeritas::orange Celeritas::testcel_harness Celeritas::testcel_core
  PRIVATE
    Celeritas::testcel_geocel Celeritas::orange ${nlohmann_json_LIBRARIES}
)

#-----------------------------------------------------------------------------#
# VARIABLES
#-----------------------------------------------------------------------------#

if(NOT CELERITAS_USE_JSON)
  set(_needs_json DISABLE)
endif()

#-----------------------------------------------------------------------------#
# SETUP
#-----------------------------------------------------------------------------#

celeritas_setup_tests(SERIAL PREFIX orange
  LINK_LIBRARIES testcel_orange Celeritas::orange
)

#-----------------------------------------------------------------------------#
# TESTS
#-----------------------------------------------------------------------------#
# Base
celeritas_add_test(BoundingBoxUtils.test.cc)
celeritas_add_test(MatrixUtils.test.cc)
celeritas_add_test(OrangeTypes.test.cc)
celeritas_add_device_test(Orange)

celeritas_add_test(detail/UniverseIndexer.test.cc)

# Bounding interval hierarchy
set(CELERITASTEST_PREFIX orange/bih)
celeritas_add_test(detail/BIHBuilder.test.cc)
celeritas_add_test(detail/BIHTraverser.test.cc)
celeritas_add_test(detail/BIHUtils.test.cc)

#-----------------------------------------------------------------------------#
# Input construction
set(CELERITASTEST_PREFIX orange/orangeinp)
celeritas_add_test(orangeinp/ConvexRegion.test.cc)
celeritas_add_test(orangeinp/ConvexSurfaceBuilder.test.cc)
celeritas_add_test(orangeinp/CsgObject.test.cc)
celeritas_add_test(orangeinp/CsgTree.test.cc)
celeritas_add_test(orangeinp/CsgTreeUtils.test.cc)
celeritas_add_test(orangeinp/Shape.test.cc)
celeritas_add_test(orangeinp/Solid.test.cc)
celeritas_add_test(orangeinp/Transformed.test.cc)
celeritas_add_test(orangeinp/UnitProto.test.cc)

celeritas_add_test(orangeinp/detail/BoundingZone.test.cc)
celeritas_add_test(orangeinp/detail/LocalSurfaceInserter.test.cc)
celeritas_add_test(orangeinp/detail/TransformInserter.test.cc)

#-----------------------------------------------------------------------------#
# Transforms
set(CELERITASTEST_PREFIX orange/transform)
celeritas_add_test(transform/SignedPermutation.test.cc)
celeritas_add_test(transform/TransformSimplifier.test.cc)
celeritas_add_test(transform/Transformation.test.cc)
celeritas_add_test(transform/Translation.test.cc)
celeritas_add_test(transform/VariantTransform.test.cc)

#-----------------------------------------------------------------------------#
# Surfaces
set(CELERITASTEST_PREFIX orange/surf)
celeritas_add_test(surf/ConeAligned.test.cc)
celeritas_add_test(surf/CylAligned.test.cc)
celeritas_add_test(surf/CylCentered.test.cc)
celeritas_add_test(surf/GeneralQuadric.test.cc)
celeritas_add_test(surf/Plane.test.cc)
celeritas_add_test(surf/PlaneAligned.test.cc)
celeritas_add_test(surf/SimpleQuadric.test.cc)
celeritas_add_test(surf/Sphere.test.cc)
celeritas_add_test(surf/SphereCentered.test.cc)

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

#-----------------------------------------------------------------------------#
# Universe details
set(CELERITASTEST_PREFIX orange/univ)
celeritas_add_test(univ/VolumeView.test.cc)
celeritas_add_test(univ/RectArrayTracker.test.cc ${_needs_json})
celeritas_add_device_test(univ/SimpleUnitTracker)
celeritas_add_test(univ/TrackerVisitor.test.cc ${_needs_json})

# Universe details
celeritas_add_test(univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(univ/detail/LogicStack.test.cc)
celeritas_add_test(univ/detail/RaggedRightIndexer.test.cc)
celeritas_add_test(univ/detail/SurfaceFunctors.test.cc)
celeritas_add_test(univ/detail/SenseCalculator.test.cc)

#-----------------------------------------------------------------------------#
