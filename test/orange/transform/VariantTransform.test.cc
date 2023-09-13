//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/VariantTransform.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/VariantTransform.hh"

#include "orange/BoundingBox.hh"
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

    VariantTransform const identity;
    EXPECT_TRUE(holds_alternative<NoTransformation>(
        apply_transform(identity, identity)));
    EXPECT_TRUE(holds_alternative<Translation>(apply_transform(identity, tl)));
    EXPECT_TRUE(
        holds_alternative<Transformation>(apply_transform(identity, tf)));

    EXPECT_TRUE(holds_alternative<Translation>(apply_transform(tl, identity)));
    EXPECT_TRUE(holds_alternative<Translation>(apply_transform(tl, tl)));
    EXPECT_TRUE(holds_alternative<Transformation>(apply_transform(tl, tf)));

    EXPECT_TRUE(
        holds_alternative<Transformation>(apply_transform(tf, identity)));
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

TEST_F(VariantTransformTest, bbox)
{
    auto bb = apply_transform(
        VariantTransform{std::in_place_type<Translation>, Real3{1, 2, 3}},
        BBox{{1, 2, 3}, {4, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({2, 4, 6}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({5, 7, 9}), bb.upper());

    bb = apply_transform(VariantTransform{std::in_place_type<Transformation>,
                                          make_rotation(Axis::z, Turn{0.25}),
                                          Real3{0, 0, 2}},
                         BBox{{1, 2, 3}, {4, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({-5, 1, 5}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({-2, 4, 8}), bb.upper());

    // IntersectionShapeTest.we_need_to_go_deeper
    bb = BBox{{-3, -3, -3}, {3, 3, 3}};  // a_sphere
    bb = apply_transform(Translation{{1, 1, 0}}, bb);  // placed_meta
    bb = apply_transform(
        Transformation{make_rotation(Axis::x, Turn{0.25}), {0, 0, 10}},
        bb);  // final bbox
    EXPECT_VEC_SOFT_EQ(Real3({-2, -3, 8}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({4, 3, 14}), bb.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
