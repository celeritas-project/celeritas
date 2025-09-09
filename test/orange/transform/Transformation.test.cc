//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/transform/Transformation.test.cc
//---------------------------------------------------------------------------//
#include "orange/transform/Transformation.hh"

#include <sstream>

#include "orange/MatrixUtils.hh"
#include "orange/transform/TransformIO.hh"
#include "orange/transform/Translation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class TransformationTest : public ::celeritas::test::Test
{
  protected:
    template<class T, size_type N>
    std::vector<T> flattened(SquareMatrix<T, N> const& inp) const
    {
        std::vector<T> result;

        for (auto& row : inp)
        {
            result.insert(result.end(), row.begin(), row.end());
        }
        return result;
    }
};

TEST_F(TransformationTest, construction)
{
    static double const identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    {
        SCOPED_TRACE("default construct");
        Transformation tr;
        EXPECT_VEC_SOFT_EQ(identity, flattened(tr.rotation()));
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), tr.translation());
    }
    {
        SCOPED_TRACE("promotion");
        Transformation tr{Translation{Real3{1, 2, 3}}};
        EXPECT_VEC_SOFT_EQ(identity, flattened(tr.rotation()));
        EXPECT_VEC_SOFT_EQ((Real3{1, 2, 3}), tr.translation());
    }
    {
        SCOPED_TRACE("serialize");
        Transformation tr{make_rotation(Axis::z, Turn{0.125}), {1, 2, 3}};
        Transformation tr2{tr.data()};
        EXPECT_EQ(tr.translation(), tr2.translation());
        EXPECT_EQ(tr.rotation(), tr2.rotation());
    }
    {
        SCOPED_TRACE("inverse");

        Transformation tr{make_rotation(Axis::z, Turn{0.125}), {1, 2, 3}};
        auto trinv
            = Transformation::from_inverse(tr.rotation(), tr.translation());

        EXPECT_VEC_SOFT_EQ(trinv.transform_down(Real3{2, -4, 0.1}),
                           tr.transform_up(Real3{2, -4, 0.1}));
        EXPECT_VEC_SOFT_EQ(
            (Real3{2, -4, 0.1}),
            trinv.transform_down(tr.transform_down(Real3{2, -4, 0.1})));

        auto trinv2 = tr.calc_inverse();
        EXPECT_EQ(trinv.translation(), trinv2.translation());
        EXPECT_EQ(trinv.rotation(), trinv2.rotation());
    }
}

TEST_F(TransformationTest, output)
{
    Transformation tr{make_rotation(Axis::x, Turn{0.25}), Real3{1, 2, 3}};
    std::ostringstream os;
    os << tr;
    EXPECT_EQ("{{{1,0,0},{0,0,-1},{0,1,0}}, {1,2,3}}", os.str());
}

TEST_F(TransformationTest, transform)
{
    {
        Transformation tr{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 1}};
        // Daughter to parent: rotate quarter turn around Z, then add 1 to Z
        EXPECT_VEC_EQ((Real3{-3, 2, 1}), tr.transform_up({2, 3, 0}));
        // Parent to daughter: subtract, then rotate back
        EXPECT_VEC_EQ((Real3{2, 3, 0}), tr.transform_down({-3, 2, 1}));

        auto props = tr.calc_properties();
        EXPECT_FALSE(props.reflects);
        EXPECT_FALSE(props.scales);
    }
    {
        Transformation tr{
            make_rotation(
                Axis::x,
                native_value_to<Turn>(std::acos(-0.5)),
                make_rotation(Axis::y, native_value_to<Turn>(std::acos(0.2)))),
            Real3{1.1, -0.5, 3.2}};

        auto const daughter = Real3{-3.4, 2.1, 0.4};
        auto const parent = tr.transform_up(daughter);
        EXPECT_VEC_NEAR((Real3{0.81191836, -4.5042777, 3.31300032}),
                        parent,
                        real_type(1e-6));
        EXPECT_VEC_SOFT_EQ(daughter, tr.transform_down(parent));
    }
}

TEST_F(TransformationTest, rotate)
{
    Transformation tr{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 1}};
    // Daughter to parent: rotate quarter turn around Z
    EXPECT_VEC_EQ((Real3{0, 1, 0}), tr.rotate_up({1, 0, 0}));
    // Parent to daughter: rotate back
    EXPECT_VEC_EQ((Real3{1, 0, 0}), tr.rotate_down({0, 1, 0}));

    auto props = tr.calc_properties();
    EXPECT_FALSE(props.reflects);
    EXPECT_FALSE(props.scales);
}

TEST_F(TransformationTest, reflect)
{
    // D2P: reflect across yz plane, then translate
    Transformation tr{make_reflection(Axis::x), Real3{1, 0, 2}};
    EXPECT_VEC_EQ((Real3{-1, 0, 0}), tr.rotate_up({1, 0, 0}));
    EXPECT_VEC_EQ((Real3{1, 0, 0}), tr.rotate_down({-1, 0, 0}));
    EXPECT_VEC_EQ((Real3{0, 2, 5}), tr.transform_up({1, 2, 3}));
    EXPECT_VEC_EQ((Real3{1, 2, 3}), tr.transform_down({0, 2, 5}));

    auto props = tr.calc_properties();
    EXPECT_TRUE(props.reflects);
    EXPECT_FALSE(props.scales);
}

TEST_F(TransformationTest, DISABLED_scale)
{
    Transformation tr{make_scaling({0.5, 1, 2}), Real3{0, 0, 0}};
    // TODO: scaling must *not* change magnitude
    EXPECT_VEC_EQ((Real3{1, 0, 0}), tr.rotate_up({1, 0, 0}));
    EXPECT_VEC_EQ((Real3{1, 0, 0}), tr.rotate_down({1, 0, 0}));
    EXPECT_VEC_EQ((Real3{0.5, 2, 6}), tr.transform_up({1, 2, 3}));
    // TODO: transformation is wrong
    EXPECT_VEC_EQ((Real3{1, 2, 3}), tr.transform_down({0.5, 2, 6}));

    auto props = tr.calc_properties();
    EXPECT_FALSE(props.reflects);
    EXPECT_TRUE(props.scales);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
