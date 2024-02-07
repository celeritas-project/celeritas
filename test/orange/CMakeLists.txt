#----------------------------------*-CMake-*----------------------------------#
# Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

celeritas_add_library(testcel_orange
  CsgTestUtils.cc
  OrangeGeoTestBase.cc
  OrangeTestBase.cc
)
celeritas_target_link_libraries(testcel_orange
  PUBLIC
    Celeritas::orange Celeritas::testcel_harness
  PRIVATE
    Celeritas::testcel_geocel Celeritas::testcel_core Celeritas::orange ${nlohmann_json_LIBRARIES}
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

# Base detail
celeritas_add_test(detail/BIHBuilder.test.cc)
celeritas_add_test(detail/BIHTraverser.test.cc)
celeritas_add_test(detail/BIHUtils.test.cc)
celeritas_add_test(detail/UniverseIndexer.test.cc)

#-----------------------------------------------------------------------------#
# Transforms
set(CELERITASTEST_PREFIX transform)
celeritas_add_test(transform/SignedPermutation.test.cc)
celeritas_add_test(transform/Transformation.test.cc)
celeritas_add_test(transform/Translation.test.cc)
celeritas_add_test(transform/VariantTransform.test.cc)

#-----------------------------------------------------------------------------#
# Surfaces
set(CELERITASTEST_PREFIX surf)
celeritas_add_test(surf/ConeAligned.test.cc)
celeritas_add_test(surf/CylAligned.test.cc)
celeritas_add_test(surf/CylCentered.test.cc)
celeritas_add_test(surf/GeneralQuadric.test.cc)
celeritas_add_test(surf/Plane.test.cc)
celeritas_add_test(surf/PlaneAligned.test.cc)
celeritas_add_test(surf/SimpleQuadric.test.cc)
celeritas_add_test(surf/Sphere.test.cc)
celeritas_add_test(surf/SphereCentered.test.cc)

celeritas_add_test(surf/FaceNamer.test.cc)
celeritas_add_test(surf/RecursiveSimplifier.test.cc)
celeritas_add_test(surf/SoftSurfaceEqual.test.cc)
celeritas_add_test(surf/SurfaceClipper.test.cc)
celeritas_add_test(surf/SurfaceSimplifier.test.cc)
celeritas_add_device_test(surf/LocalSurfaceVisitor)

celeritas_add_test(surf/detail/QuadraticSolver.test.cc)
celeritas_add_test(surf/detail/SurfaceTranslator.test.cc)
celeritas_add_test(surf/detail/SurfaceTransformer.test.cc)

#-----------------------------------------------------------------------------#
# Construct
set(CELERITASTEST_PREFIX construct)
celeritas_add_test(construct/CsgTree.test.cc)
celeritas_add_test(construct/CsgTreeUtils.test.cc)
celeritas_add_test(construct/LocalSurfaceInserter.test.cc)

set(CELERITASTEST_PREFIX orangeinp)
celeritas_add_test(orangeinp/detail/BoundingZone.test.cc)
celeritas_add_test(orangeinp/detail/CsgUnitBuilder.test.cc)

#-----------------------------------------------------------------------------#
# Universe details
set(CELERITASTEST_PREFIX univ)
celeritas_add_test(univ/detail/LogicEvaluator.test.cc)
celeritas_add_test(univ/detail/LogicStack.test.cc)
celeritas_add_test(univ/detail/RaggedRightIndexer.test.cc)
celeritas_add_test(univ/detail/SurfaceFunctors.test.cc)
celeritas_add_test(univ/detail/SenseCalculator.test.cc)
celeritas_add_test(univ/VolumeView.test.cc)
celeritas_add_test(univ/RectArrayTracker.test.cc ${_needs_json})
celeritas_add_device_test(univ/SimpleUnitTracker)
celeritas_add_test(univ/TrackerVisitor.test.cc ${_needs_json})

#-----------------------------------------------------------------------------#
