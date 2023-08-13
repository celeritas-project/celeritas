//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/BoundingBoxUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/BoundingBoxUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
class BoundingBoxUtilsTest : public Test
{
};

TEST_F(BoundingBoxUtilsTest, is_inside)
{
    BBox bbox1 = {{-5, -2, -100}, {6, 1, 1}};
    EXPECT_TRUE(is_inside(bbox1, Real3{-4, 0, 0}));
    EXPECT_TRUE(is_inside(bbox1, Real3{-4.9, -1.9, -99.9}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-6, 0, 0}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-5.1, -2.1, -101.1}));
}

TEST_F(BoundingBoxUtilsTest, is_infinite)
{
    auto inf_real = std::numeric_limits<real_type>::infinity();

    BBox bbox1 = {{0, 0, 0}, {1, 1, 1}};
    EXPECT_FALSE(is_infinite(bbox1));

    BBox bbox2 = {{0, 0, 0}, {inf_real, inf_real, inf_real}};
    EXPECT_FALSE(is_infinite(bbox2));

    BBox bbox3
        = {{-inf_real, -inf_real, -inf_real}, {inf_real, inf_real, inf_real}};
    EXPECT_TRUE(is_infinite(bbox3));
}

TEST_F(BoundingBoxUtilsTest, center)
{
    BBox bbox = {{-10, -20, -30}, {1, 2, 3}};
    EXPECT_VEC_SOFT_EQ(Real3({-4.5, -9, -13.5}), calc_center(bbox));
}

TEST_F(BoundingBoxUtilsTest, surface_area)
{
    BBox bbox = {{-1, -2, -3}, {6, 4, 5}};
    EXPECT_SOFT_EQ(2 * (7 * 6 + 7 * 8 + 6 * 8), calc_surface_area(bbox));
}

TEST_F(BoundingBoxUtilsTest, bbox_union)
{
    auto ubox = calc_union(BBox{{-10, -20, -30}, {10, 2, 3}},
                           BBox{{-15, -9, -33}, {1, 2, 10}});

    EXPECT_VEC_SOFT_EQ(Real3({-15, -20, -33}), ubox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({10, 2, 10}), ubox.upper());

    {
        SCOPED_TRACE("degenerate");
        auto dubox = calc_union(ubox, BBox{});
        EXPECT_VEC_SOFT_EQ(ubox.lower(), dubox.lower());
        EXPECT_VEC_SOFT_EQ(ubox.upper(), dubox.upper());
    }
    {
        SCOPED_TRACE("double degenerate");
        auto ddbox = calc_union(BBox{}, BBox{});
        EXPECT_FALSE(ddbox);
    }
}

TEST_F(BoundingBoxUtilsTest, bbox_intersection)
{
    auto ibox = calc_intersection(BBox{{-10, -20, -30}, {10, 2, 3}},
                                  BBox{{-15, -9, -33}, {1, 2, 10}});

    EXPECT_VEC_SOFT_EQ(Real3({-10, -9, -30}), ibox.lower());
    EXPECT_VEC_SOFT_EQ(Real3({1, 2, 3}), ibox.upper());

    {
        SCOPED_TRACE("degenerate");
        auto dibox = calc_intersection(ibox, BBox{});
        EXPECT_FALSE(dibox);
        EXPECT_VEC_SOFT_EQ(BBox{}.lower(), dibox.lower());
        EXPECT_VEC_SOFT_EQ(BBox{}.upper(), dibox.upper());
    }
    {
        SCOPED_TRACE("double degenerate");
        auto ddbox = calc_intersection(BBox{}, BBox{});
        EXPECT_FALSE(ddbox);
    }
    {
        SCOPED_TRACE("partial degenerate x");
        auto dibox = calc_intersection(BBox{{-2, -inf, -inf}, {2, inf, inf}},
                                       BBox{{3, -inf, -inf}, {10, inf, inf}});
        EXPECT_FALSE(dibox);
    }
    {
        SCOPED_TRACE("partial degenerate y");
        auto dibox = calc_intersection(BBox{{-inf, -2, -inf}, {inf, 2, inf}},
                                       BBox{{-inf, 3, -inf}, {inf, 10, inf}});
        EXPECT_FALSE(dibox);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
