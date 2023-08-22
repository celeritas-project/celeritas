//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/VariantTransform.hh"

#include "orange/transform/Transformation.hh"
#include "orange/transform/Translation.hh"
#include "orange/transform/detail/TransformTransformer.hh"
#include "orange/transform/detail/TransformTranslator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class VariantTransformTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(VariantTransformTest, translator)
{
    Translation upper{Real3{0, 0, 1}};
    detail::TransformTranslator apply(upper);
    {
        Transformation lower{make_rotation(Axis::x, Turn{0.5}), Real3{3, 0, 0}};
        auto combined = apply(lower);
        EXPECT_VEC_SOFT_EQ((Real3{3, 0, 1}), combined.translation());

        Real3 pos{0.5, 1.5, 2.5};
        EXPECT_VEC_SOFT_EQ(upper.transform_up(lower.transform_up(pos)),
                           combined.transform_up(pos));
    }
    {
        Translation lower{Real3{3, 0, 0}};
        auto combined = apply(lower);
        EXPECT_VEC_SOFT_EQ((Real3{3, 0, 1}), combined.translation());

        Real3 pos{0.5, 1.5, 2.5};
        EXPECT_VEC_SOFT_EQ(upper.transform_up(lower.transform_up(pos)),
                           combined.transform_up(pos));
    }
}

TEST_F(VariantTransformTest, transformer)
{
    Transformation upper{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 1}};
    detail::TransformTransformer apply(upper);
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

TEST_F(VariantTransformTest, variant_types)
{
    using std::holds_alternative;
    Translation tl{Real3{0, 1, 0}};
    Transformation tf{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 2}};

    EXPECT_TRUE(holds_alternative<std::monostate>(apply_transform({}, {})));
    EXPECT_TRUE(holds_alternative<Translation>(apply_transform({}, tl)));
    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform({}, tf)));

    EXPECT_TRUE(holds_alternative<Translation>(apply_transform(tl, {})));
    EXPECT_TRUE(holds_alternative<Translation>(apply_transform(tl, tl)));
    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform(tl, tf)));

    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform(tf, {})));
    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform(tf, tl)));
    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform(tf, tf)));

    auto result = apply_transform(tl, tf);
    if (auto* result_tf = std::get_if<Transformation>(&result))
    {
        EXPECT_EQ(result_tf->rotation(), tf.rotation());
        EXPECT_VEC_SOFT_EQ((Real3{0, 1, 2}), result_tf->translation());
    }
    else
    {
        FAIL() << "wrong type";
    }

    Transformation tf2{make_rotation(Axis::x, Turn{0.3}), Real3{1, 0, 2}};
    result = apply_transform(tf, tf2);
    if (auto* result_tf = std::get_if<Transformation>(&result))
    {
        Real3 pos{1, 2, 3};
        EXPECT_VEC_SOFT_EQ(tf.transform_up(tf2.transform_up(pos)),
                           result_tf->transform_up(pos));
    }
    else
    {
        FAIL() << "wrong type";
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
