//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/TransformSimplifier.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/TransformSimplifier.hh"

#include <cmath>

#include "corecel/math/ArrayUtils.hh"
#include "orange/MatrixUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct DataGetter
{
    template<class T>
    Span<real_type const> operator()(T const& tf) const
    {
        return tf.data();
    }
};

// Compare a known transformation against an unknown
template<class T, class U>
::testing::AssertionResult IsTransformEq(char const* expected_expr,
                                         char const* actual_expr,
                                         T const& expected,
                                         U const& actual)
{
    DataGetter const get_data;
    return ::celeritas::testdetail::IsVecSoftEquiv(
        expected_expr,
        actual_expr,
        get_data(expected),
        std::visit(get_data, actual));
}

#define EXPECT_TR_SOFT_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(IsTransformEq, expected, actual)

//---------------------------------------------------------------------------//
class TransformSimplifierTest : public ::celeritas::test::Test
{
  protected:
    using Tol = Tolerance<>;

    Tol tol_ = Tol::from_relative(1e-4);
};

TEST_F(TransformSimplifierTest, notransform)
{
    TransformSimplifier simplify{tol_};
    EXPECT_TR_SOFT_EQ(NoTransformation{}, simplify(NoTransformation{}));
}

TEST_F(TransformSimplifierTest, translate)
{
    TransformSimplifier simplify{tol_};

    EXPECT_TR_SOFT_EQ(NoTransformation{}, simplify(Translation{{0, 0, 0}}));
    EXPECT_TR_SOFT_EQ(NoTransformation{}, simplify(Translation{{1e-5, 0, 0}}));

    Translation micro{{1e-4, 0, 1e-4}};
    EXPECT_TR_SOFT_EQ(micro, simplify(micro));

    Translation macro{{0, 0, 1}};
    EXPECT_TR_SOFT_EQ(macro, simplify(macro));
}

TEST_F(TransformSimplifierTest, transform)
{
    real_type const eps_theta = 2 * std::asin(tol_.abs / 2);
    auto micro_r = make_rotation(Axis::x, Turn{1e-8});
    auto tiny_r = make_rotation(make_unit_vector(Real3{3, -4, 5}),
                                native_value_to<Turn>(eps_theta / 2));
    auto large_r = make_rotation(make_unit_vector(Real3{3, -4, 5}),
                                 native_value_to<Turn>(eps_theta * 2));
    TransformSimplifier simplify{tol_};

    {
        SCOPED_TRACE("simplify nothing");
        Transformation const orig(large_r, Real3{0, 1e-2, 0});
        auto actual = simplify(orig);
        EXPECT_TR_SOFT_EQ(orig, actual);
    }
    {
        SCOPED_TRACE("simplify to rotation");
        auto actual = simplify(Transformation(large_r, Real3{0, 0, 0}));
        EXPECT_TR_SOFT_EQ(Transformation(large_r, Real3{0, 0, 0}), actual);
    }
    {
        SCOPED_TRACE("simplify to translation (micro)");
        auto actual = simplify(Transformation(micro_r, Real3{0, 1, 0}));
        EXPECT_TR_SOFT_EQ(Translation(Real3{0, 1, 0}), actual);
    }
    {
        SCOPED_TRACE("simplify to translation (tiny)");
        auto actual = simplify(Transformation(tiny_r, Real3{1, 0, 1}));
        EXPECT_TR_SOFT_EQ(Translation(Real3{1, 0, 1}), actual);
    }
    {
        SCOPED_TRACE("simplify to notrans");
        auto actual = simplify(Transformation(tiny_r, Real3{0, 1e-5, 0}));
        EXPECT_TR_SOFT_EQ(NoTransformation{}, actual);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
