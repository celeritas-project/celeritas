//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformTransformer.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/TransformTransformer.hh"

#include "orange/transform/Transformation.hh"
#include "orange/transform/Translation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class TransformTransformerTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(TransformTransformerTest, all)
{
    Transformation upper{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 1}};
    TransformTransformer apply(upper);
    {
        Transformation lower{make_rotation(Axis::x, Turn{0.5}), Real3{3, 0, 0}};
        auto combined = apply(lower);
        EXPECT_VEC_SOFT_EQ((Real3{0, 3, 1}), combined.translation());

        Real3 pos{0.5, 1.5, 2.5};
        EXPECT_VEC_SOFT_EQ(upper.transform_up(lower.transform_up(pos)),
                           combined.transform_up(pos));
    }
    {
        Translation lower{Real3{3, 0, 0}};
        auto combined = apply(lower);
        EXPECT_VEC_SOFT_EQ((Real3{0, 3, 1}), combined.translation());

        Real3 pos{0.5, 1.5, 2.5};
        EXPECT_VEC_SOFT_EQ(upper.transform_up(lower.transform_up(pos)),
                           combined.transform_up(pos));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
