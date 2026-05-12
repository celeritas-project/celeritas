//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBoxUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/BoundingBoxUtils.hh"

#include <limits>

#include "corecel/Types.hh"
#include "corecel/math/ArrayUtils.hh"
#include "orange/MatrixUtils.hh"
#include "orange/transform/Transformation.hh"
#include "orange/transform/Translation.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using BoundingBoxUtilsTest = Test;

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
    auto inf = std::numeric_limits<real_type>::infinity();

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

using IntersectsSegmentTest = Test;

TEST_F(IntersectsSegmentTest, basic)
{
    BBox const bbox{{0., 0., 0.}, {1, 1, 1}};

    // Basic case: pos outside by 0.1 along x
    Real3 pos{1.1, 0.5, 0.5};
    Real3 dir{-1, 0, 0};
    EXPECT_TRUE(intersects_segment(bbox, pos, dir, 0.2_r));
    EXPECT_FALSE(intersects_segment(bbox, pos, dir, 0.05_r));

    // Coming in from an angle (entry dist = 0.1 * sqrt(2_r))
    dir = Real3(-std::sqrt(2) / 2, -std::sqrt(2) / 2, 0);
    EXPECT_TRUE(intersects_segment(bbox, pos, dir, 0.2_r));
    EXPECT_FALSE(intersects_segment(bbox, pos, dir, 0.1_r));

    // First intersection point occurs outside box, but second intersection
    // point is valid (entry dist = 2 * sqrt(2_r))
    pos = Real3{3, 2.5, 0.5};
    EXPECT_TRUE(intersects_segment(bbox, pos, dir, 3.0_r));
    EXPECT_FALSE(intersects_segment(bbox, pos, dir, 2.0_r));

    // No intersection
    dir = Real3{0, -1, 0};
    EXPECT_FALSE(intersects_segment(bbox, pos, dir, 1e6_r));

    // Already inside: always true
    pos = Real3{0.5, 0.6, 0.7};
    EXPECT_TRUE(intersects_segment(bbox, pos, dir, 0.1_r));

    // Start exactly on bbox, exiting
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{1, 0, 0}, Real3{1, 0, 0}, 0.1_r));
    // Start exactly on bbox, entering
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{1, 0, 0}, Real3{-1, 0, 0}, 0.1_r));
    // End exactly on bbox, exiting
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{0.5, 0, 0}, Real3{1, 0, 0}, 0.5_r));
    // End exactly on bbox, entering
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{1.5, 0, 0}, Real3{-1, 0, 0}, 0.5_r));
}

TEST_F(IntersectsSegmentTest, near_degenerate)
{
    auto bbox = BBox{{0., 0., 0.}, {1, 1, 1}};

    // Degenerate near-parallel cases: sweep over inside/outside start,
    // inward/outward direction, and very short/very long segment lengths.
    real_type eps = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
                         ? 1e-12_r
                         : 1e-6_r);
    struct
    {
        char const* label;
        Real3 pos;
        bool is_inside;
    } const positions[] = {
        {"barely inside", {1 - eps, 0.5, 0.5}, true},
        {"barely outside", {1 + eps, 0.5, 0.5}, false},
    };

    real_type tilt
        = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e-9_r : 5e-4_r);
    real_type const entry_dist = eps / tilt;
    struct
    {
        char const* label;
        Real3 dir;
        bool is_inward;
    } const directions[] = {
        {"inward", {-tilt, std::sqrt(1 - ipow<2>(tilt)), 0}, true},
        {"parallel", {0, 1, 0}, false},
        {"outward", {tilt, std::sqrt(1 - ipow<2>(tilt)), 0}, false},
    };

    constexpr real_type large_dist
        = (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e10_r : 1e5_r);
    for (auto const max_dist : {1 / large_dist, large_dist})
    {
        for (auto const& p : positions)
        {
            for (auto const& d : directions)
            {
                SCOPED_TRACE(::testing::Message{} << p.label << ", " << d.label
                                                  << ", max_dist=" << max_dist);

                bool const expected
                    = p.is_inside || (d.is_inward && max_dist >= entry_dist);

                EXPECT_EQ(expected,
                          intersects_segment(bbox, p.pos, d.dir, max_dist));
            }
        }
    }
}

TEST_F(IntersectsSegmentTest, infinite)
{
    BBox const bbox{{0., 0., 0.}, {1, 1, 1}};

    constexpr real_type infr = std::numeric_limits<real_type>::infinity();

    // Actually intersecting
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{0.5, 0.5, 0.5}, Real3{1, 0, 0}, infr));
    EXPECT_TRUE(intersects_segment(
        bbox, Real3{0.5, 1.25, 0.5}, make_unit_vector(Real3{1, -1, 0}), infr));

    // False positives
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{1.5, 0.5, 0.5}, Real3{1, 0, 0}, infr));
    EXPECT_TRUE(
        intersects_segment(bbox, Real3{-0.5, 1.1, 0.5}, Real3{1, 0, 0}, infr));
    EXPECT_TRUE(intersects_segment(
        bbox, Real3{1.5, 0.75, 0.5}, make_unit_vector(Real3{1, -1, 0}), infr));
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
