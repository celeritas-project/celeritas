//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/TransformInserter.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/TransformInserter.hh"

#include "orange/MatrixUtils.hh"
#include "orange/transform/VariantTransform.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
//---------------------------------------------------------------------------//

class TransformInserterTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(TransformInserterTest, all)
{
    std::vector<VariantTransform> transforms;
    TransformInserter insert(&transforms);

    EXPECT_EQ(TransformId{0},
              insert(VariantTransform{std::in_place_type<NoTransformation>}));
    EXPECT_EQ(TransformId{0},
              insert(VariantTransform{std::in_place_type<NoTransformation>}));
    EXPECT_EQ(TransformId{1},
              insert(VariantTransform{std::in_place_type<Translation>,
                                      Real3{1, 2, 3}}));
    EXPECT_EQ(TransformId{2},
              insert(VariantTransform{std::in_place_type<Transformation>,
                                      make_rotation(Axis::z, Turn{0}),
                                      Real3{1, 2, 3}}));
    EXPECT_EQ(TransformId{1},
              insert(VariantTransform{std::in_place_type<Translation>,
                                      Real3{1, 2, 3}}));
    EXPECT_EQ(TransformId{0},
              insert(VariantTransform{std::in_place_type<NoTransformation>}));

    EXPECT_EQ(3, transforms.size());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
