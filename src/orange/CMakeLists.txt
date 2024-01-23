#----------------------------------*-CMake-*----------------------------------#
# Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#-----------------------------------------------------------------------------#

set(SOURCES)
set(PRIVATE_DEPS)
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
  construct/CsgTree.cc
  construct/CsgTypes.cc
  construct/CsgTreeUtils.cc
  construct/DepthCalculator.cc
  construct/detail/LocalSurfaceInserter.cc
  construct/detail/NodeSimplifier.cc
  detail/BIHBuilder.cc
  detail/BIHPartitioner.cc
  detail/RectArrayInserter.cc
  detail/SurfacesRecordBuilder.cc
  detail/UnitInserter.cc
  detail/UniverseInserter.cc
  orangeinp/detail/CsgUnitBuilder.cc
  surf/ConeAligned.cc
  surf/CylAligned.cc
  surf/FaceNamer.cc
  surf/GeneralQuadric.cc
  surf/Plane.cc
  surf/PlaneAligned.cc
  surf/SimpleQuadric.cc
  surf/Sphere.cc
  surf/SoftSurfaceEqual.cc
  surf/SurfaceIO.cc
  surf/SurfaceSimplifier.cc
  surf/VariantSurface.cc
  surf/detail/SurfaceClipperImpl.cc
  surf/detail/SurfaceTranslator.cc
  surf/detail/SurfaceTransformer.cc
  transform/SignedPermutation.cc
  transform/TransformIO.cc
  transform/Transformation.cc
  transform/VariantTransform.cc
)

if(CELERITAS_USE_JSON)
  list(APPEND SOURCES
    BoundingBoxIO.json.cc
    construct/CsgTreeIO.json.cc
    construct/OrangeInputIO.json.cc
    construct/detail/OrangeInputIOImpl.json.cc
  )
  list(APPEND PRIVATE_DEPS nlohmann_json::nlohmann_json)
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
