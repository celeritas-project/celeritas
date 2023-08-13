//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBox.test.cc
//---------------------------------------------------------------------------//
#include "orange/BoundingBox.hh"

#include <limits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using BoundingBoxTest = Test;

TEST_F(BoundingBoxTest, null)
{
    BBox null_bbox;
    EXPECT_FALSE(null_bbox);
    EXPECT_GT(null_bbox.lower()[0], null_bbox.upper()[0]);

    constexpr auto dumb_bbox = BBox::from_unchecked({3, 0, 0}, {-1, 0, 0});
    EXPECT_FALSE(dumb_bbox);

    BBox ibb = BBox::from_infinite();
    ibb.clip(BBox::Halfspace{Sense::outside, Axis::x, 2});
    ibb.clip(BBox::Halfspace{Sense::inside, Axis::x, 1});
    EXPECT_FALSE(ibb);
}

TEST_F(BoundingBoxTest, infinite)
{
    BBox ibb = BBox::from_infinite();
    EXPECT_TRUE(ibb);
    EXPECT_SOFT_EQ(-inf, ibb.lower()[0]);
    EXPECT_SOFT_EQ(-inf, ibb.lower()[1]);
    EXPECT_SOFT_EQ(-inf, ibb.lower()[2]);
    EXPECT_SOFT_EQ(inf, ibb.upper()[0]);
    EXPECT_SOFT_EQ(inf, ibb.upper()[1]);
    EXPECT_SOFT_EQ(inf, ibb.upper()[2]);
}

TEST_F(BoundingBoxTest, standard)
{
    if (CELERITAS_DEBUG)
    {
        const Real3 lo{-1, -2, -3};
        EXPECT_THROW((BBox{lo, {-4, 5, 6}}), DebugError);
        EXPECT_THROW((BBox{lo, {4, -5, 6}}), DebugError);
        EXPECT_THROW((BBox{lo, {4, 5, -6}}), DebugError);
    }

    BBox bb{{-1, -2, 3}, {4, 5, 6}};
    EXPECT_TRUE(bb);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, 3}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, 5, 6}), bb.upper());

    bb.clip(BBox::Halfspace{Sense::inside, Axis::x, 2});
    bb.clip(BBox::Halfspace{Sense::outside, Axis::z, 4});
    bb.clip(BBox::Halfspace{Sense::inside, Axis::y, 0});
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, 4}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 0, 6}), bb.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
