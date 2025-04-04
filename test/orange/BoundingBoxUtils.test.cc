//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/BoundingBoxUtils.hh"

#include "orange/MatrixUtils.hh"
#include "orange/transform/Transformation.hh"
#include "orange/transform/Translation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
class BoundingBoxUtilsTest : public Test
{
};

TEST_F(BoundingBoxUtilsTest, is_infinite)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    EXPECT_FALSE(is_infinite(BBox{{0, 0, 0}, {1, 1, 1}}));
    EXPECT_FALSE(is_infinite(BBox{{0, 0, 0}, {inf, inf, inf}}));
    EXPECT_FALSE(is_infinite(BBox{{0, -inf, -inf}, {1, inf, inf}}));
    EXPECT_FALSE(is_infinite(BBox{{inf, inf, inf}, {inf, inf, inf}}));

    EXPECT_TRUE(is_infinite(BBox{{-inf, -inf, -inf}, {inf, inf, inf}}));
}

TEST_F(BoundingBoxUtilsTest, is_finite)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    EXPECT_TRUE(is_finite(BBox{{0, 0, 0}, {1, 1, 1}}));
    EXPECT_FALSE(is_finite(BBox{{0, 0, 0}, {inf, inf, inf}}));
    EXPECT_FALSE(is_finite(BBox{{0, -inf, -inf}, {1, inf, inf}}));
    EXPECT_FALSE(is_finite(BBox{{inf, inf, inf}, {inf, inf, inf}}));

    EXPECT_FALSE(is_finite(BBox{{-inf, -inf, -inf}, {inf, inf, inf}}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(is_finite(BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, is_degenerate)
{
    EXPECT_FALSE(is_degenerate(BBox{{0, 0, 0}, {1, 1, 1}}));
    EXPECT_TRUE(is_degenerate(BBox{{0, 0, 1}, {1, 1, 1}}));
    EXPECT_TRUE(is_degenerate(BBox{{1, 0, 1}, {1, 1, 1}}));
    EXPECT_TRUE(is_degenerate(BBox{{1, 1, 1}, {1, 1, 1}}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(is_degenerate(BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, center)
{
    BBox bbox = {{-10, -20, -30}, {1, 2, 3}};
    EXPECT_VEC_SOFT_EQ(Real3({-4.5, -9, -13.5}), calc_center(bbox));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_center(BBox{}), DebugError);
    }

    bbox = BBox({{-10, -20, -inf}, {1, 2, inf}});
    EXPECT_VEC_SOFT_EQ(Real3({-4.5, -9, 0}), calc_center(bbox));

    if (CELERITAS_DEBUG)
    {
        bbox = BBox({{-10, -20, 5}, {1, 2, inf}});
        EXPECT_THROW(calc_center(bbox), DebugError);

        bbox = BBox({{-10, -20, -inf}, {1, 2, 5}});
        EXPECT_THROW(calc_center(bbox), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, half_widths)
{
    BBox bbox = {{-10, -20, -30}, {1, 2, 3}};
    EXPECT_VEC_SOFT_EQ(Real3({5.5, 11, 16.5}), calc_half_widths(bbox));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_half_widths(BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, surface_area)
{
    BBox bbox = {{-1, -2, -3}, {6, 4, 5}};
    EXPECT_SOFT_EQ(2 * (7 * 6 + 7 * 8 + 6 * 8), calc_surface_area(bbox));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_surface_area(BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, volume)
{
    BBox bbox = {{-1, -2, -3}, {6, 4, 5}};
    EXPECT_SOFT_EQ(7 * 6 * 8, calc_volume(bbox));

    // Degenerate volume
    EXPECT_SOFT_EQ(0, calc_volume(BBox{{1, 1, 1}, {1, 2, 3}}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_volume(BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_union)
{
    auto ubox = calc_union(BBox{{-10, -20, -30}, {10, 2, 3}},
                           BBox{{-15, -9, -33}, {1, 2, 10}});

    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -33}), ubox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), ubox.upper());

    {
        SCOPED_TRACE("null");
        auto dubox = calc_union(ubox, BBox{});
        EXPECT_VEC_SOFT_EQ(ubox.lower(), dubox.lower());
        EXPECT_VEC_SOFT_EQ(ubox.upper(), dubox.upper());
    }
    {
        SCOPED_TRACE("double null");
        auto ddbox = calc_union(BBox{}, BBox{});
        EXPECT_FALSE(ddbox);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_intersection)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    auto ibox = calc_intersection(BBox{{-10, -20, -30}, {10, 2, 3}},
                                  BBox{{-15, -9, -33}, {1, 2, 10}});

    EXPECT_VEC_SOFT_EQ(Real3({-10, -9, -30}), ibox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({1, 2, 3}), ibox.upper());

    {
        SCOPED_TRACE("nonintersecting is null");
        auto nbox = calc_intersection(BBox{{-1, -1, -1}, {1, 1, 1}},
                                      BBox{{1.1, 0, 0}, {2, 1, 1}});
        EXPECT_FALSE(nbox);
    }
    {
        SCOPED_TRACE("common point/line/face is degenerate");
        auto dbox = calc_intersection(BBox{{-1, -1, -1}, {1, 1, 1}},
                                      BBox{{-1, -1, 1}, {2, 2, 2}});
        EXPECT_TRUE(dbox);
        EXPECT_VEC_SOFT_EQ(Real3({-1, -1, 1}), dbox.lower());
        EXPECT_VEC_SOFT_EQ(Real3({1, 1, 1}), dbox.upper());
    }
    {
        SCOPED_TRACE("null");
        auto dibox = calc_intersection(ibox, BBox{});
        EXPECT_FALSE(dibox);
        EXPECT_VEC_SOFT_EQ(BBox{}.lower(), dibox.lower());
        EXPECT_VEC_SOFT_EQ(BBox{}.upper(), dibox.upper());
    }
    {
        SCOPED_TRACE("double null");
        auto ddbox = calc_intersection(BBox{}, BBox{});
        EXPECT_FALSE(ddbox);
    }
    {
        SCOPED_TRACE("partial null x");
        auto dibox = calc_intersection(BBox{{-2, -inf, -inf}, {2, inf, inf}},
                                       BBox{{3, -inf, -inf}, {10, inf, inf}});
        EXPECT_FALSE(dibox);
    }
    {
        SCOPED_TRACE("partial null y");
        auto dibox = calc_intersection(BBox{{-inf, -2, -inf}, {inf, 2, inf}},
                                       BBox{{-inf, 3, -inf}, {inf, 10, inf}});
        EXPECT_FALSE(dibox);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_dist_to_inside)
{
    using Real3 = Array<double, 3>;

    auto bbox = BBox{{0., 0., 0.}, {1, 1, 1}};

    // Basic case
    Real3 pos{1.1, 0.5, 0.5};
    Real3 dir{-1, 0, 0};
    EXPECT_SOFT_EQ(0.1, calc_dist_to_inside(bbox, pos, dir));

    // Coming in from an angle
    dir = Real3{-std::sqrt(2) / 2, -std::sqrt(2) / 2, 0};
    EXPECT_SOFT_EQ(0.1 * std::sqrt(2), calc_dist_to_inside(bbox, pos, dir));

    // First intersection point occurs outside box, but second intersection
    // point is valid
    pos = Real3{3, 2.5, 0.5};
    EXPECT_SOFT_EQ(2 * std::sqrt(2), calc_dist_to_inside(bbox, pos, dir));

    // No intersection
    dir = Real3{0, -1, 0};
    EXPECT_EQ(numeric_limits<double>::infinity(),
              calc_dist_to_inside(bbox, pos, dir));

    // Already inside
    if (CELERITAS_DEBUG)
    {
        pos = Real3{0.5, 0.6, 0.7};
        EXPECT_THROW(calc_dist_to_inside(bbox, pos, dir), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_encloses)
{
    EXPECT_TRUE(encloses(BBox::from_infinite(), BBox{{-1, -1, -1}, {1, 1, 1}}));
    EXPECT_TRUE(encloses(BBox{{-1, -2, -3}, {2, 2, 3}},
                         BBox{{-1, -1, -1}, {1, 1, 1}}));
    EXPECT_FALSE(encloses(BBox{{-1, -2, -3}, {1, 1.5, 3}},
                          BBox{{-1, -2, -3}, {1, 2, 3}}));

    EXPECT_FALSE(encloses(BBox{}, BBox{{-1, -2, -3}, {1, 2, 3}}));
    EXPECT_TRUE(encloses(BBox{{-1, -2, -3}, {1, 2, 3}}, BBox{}));
    EXPECT_TRUE(encloses(BBox::from_infinite(), BBox{}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(encloses(BBox{}, BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, bumped)
{
    auto inf = std::numeric_limits<real_type>::infinity();
    BoundingBox<double> const ref{{-inf, 0, -100}, {0, 0.11223344556677, inf}};

    {
        SCOPED_TRACE("default precision");
        BoundingBoxBumper<float, double> calc_bumped{};
        auto bumped = calc_bumped(ref);
        static float const expected_lower[] = {-inff, -1e-14f, -100.0001f};
        static float const expected_upper[] = {1e-14f, 0.1122335f, inff};
        EXPECT_VEC_SOFT_EQ(expected_lower, bumped.lower());
        EXPECT_VEC_SOFT_EQ(expected_upper, bumped.upper());

        EXPECT_TRUE(is_inside(bumped, ref.lower()));
        EXPECT_TRUE(is_inside(bumped, ref.upper()));
    }
    {
        SCOPED_TRACE("double precise");
        BoundingBoxBumper<double> calc_bumped{
            Tolerance<double>::from_relative(1e-10)};
        auto bumped = calc_bumped(ref);
        static double const expected_lower[] = {-inf, -1e-10, -100.00000001};
        static double const expected_upper[] = {1e-10, 0.11223344566677, inf};
        EXPECT_VEC_SOFT_EQ(expected_lower, bumped.lower());
        EXPECT_VEC_SOFT_EQ(expected_upper, bumped.upper());

        EXPECT_TRUE(is_inside(bumped, ref.lower() - 1e-11));
        EXPECT_TRUE(is_inside(bumped, ref.upper() + 1e-11));
    }
    {
        SCOPED_TRACE("float loose");
        BoundingBoxBumper<float, double> calc_bumped{
            Tolerance<double>::from_relative(1e-3, /* length = */ 0.01)};
        auto bumped = calc_bumped(ref);
        static float const expected_lower[] = {-inff, -1e-05f, -100.1f};
        static float const expected_upper[] = {1e-05f, 0.1123457f, inff};
        EXPECT_VEC_SOFT_EQ(expected_lower, bumped.lower());
        EXPECT_VEC_SOFT_EQ(expected_upper, bumped.upper());

        EXPECT_TRUE(is_inside(bumped, ref.lower() - 1e-6));
        EXPECT_TRUE(is_inside(bumped, ref.upper() + 1e-6));
    }
    {
        SCOPED_TRACE("float orange");
        BoundingBox<double> const ref{{-2, -6, -1}, {8, 4, 2}};
        static_assert(std::is_same_v<decltype(ref)::real_type, double>);
        BoundingBoxBumper<float, double> calc_bumped{
            Tolerance<double>::from_relative(2e-8)};
        auto bumped = calc_bumped(ref);
        static float const expected_lower[] = {-2.f, -6.f, -1.f};
        static float const expected_upper[] = {8.000001f, 4.f, 2.f};
        EXPECT_VEC_SOFT_EQ(expected_lower, bumped.lower());
        EXPECT_VEC_SOFT_EQ(expected_upper, bumped.upper());

        EXPECT_TRUE(is_inside(bumped, ref.lower() - 1e-8));
        EXPECT_TRUE(is_inside(bumped, ref.upper() + 1e-8));
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_translate)
{
    Translation const tr{{1, 2, 3}};
    auto bb = calc_transform(tr, BBox{{1, 2, 3}, {4, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({2, 4, 6}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({5, 7, 9}), bb.upper());

    bb = calc_transform(tr, BBox::from_infinite());
    EXPECT_EQ(bb.lower(), BBox::from_infinite().lower());
    EXPECT_EQ(bb.upper(), BBox::from_infinite().upper());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_transform(tr, BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_transform)
{
    auto inf = std::numeric_limits<real_type>::infinity();

    // Daughter to parent: rotate quarter turn around Z, then add 1 to Z
    Transformation tr{make_rotation(Axis::z, Turn{0.25}), Real3{0, 0, 1}};

    auto bb = calc_transform(tr, BBox{{1, 2, 3}, {4, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({-5, 1, 4}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({-2, 4, 7}), bb.upper());

    // Test infinities
    bb = calc_transform(tr, BBox{{-inf, 2, 3}, {inf, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({-5, -inf, 4}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({-2, inf, 7}), bb.upper());

    // Transform again
    bb = calc_transform(tr, bb);
    EXPECT_VEC_SOFT_EQ(Real3({-inf, -5, 5}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({inf, -2, 8}), bb.upper());

    // Transform a part of a turn that results in rotated but still infinite
    // space
    bb = calc_transform(
        Transformation{make_rotation(Axis::z, Turn{0.001}), Real3{0, 0, 0}},
        BBox{{-inf, 2, 3}, {inf, 5, 6}});
    EXPECT_VEC_SOFT_EQ(Real3({-inf, -inf, 3}), bb.lower());
    EXPECT_VEC_SOFT_EQ(Real3({inf, inf, 6}), bb.upper());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_transform(tr, BBox{}), DebugError);
    }
}

TEST_F(BoundingBoxUtilsTest, stream)
{
    {
        std::ostringstream os;
        os << BBox{};
        EXPECT_EQ("{}", os.str());
    }
    {
        std::ostringstream os;
        os << BBox{{1, 2, 3}, {4, 5, 6}};
        EXPECT_EQ("{{1,2,3}, {4,5,6}}", os.str());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
