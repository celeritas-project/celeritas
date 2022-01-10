//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BoundingBox.test.cc
//---------------------------------------------------------------------------//
#include "geometry/BoundingBox.hh"

#include "celeritas_test.hh"

using celeritas::BoundingBox;
using celeritas::Real3;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class BoundingBoxTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(BoundingBoxTest, null)
{
    BoundingBox null_bbox;
    EXPECT_FALSE(null_bbox);
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(null_bbox.lower(), celeritas::DebugError);
        EXPECT_THROW(null_bbox.upper(), celeritas::DebugError);
    }
}

TEST_F(BoundingBoxTest, infinite)
{
    BoundingBox ibb = BoundingBox::from_infinite();
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
        EXPECT_THROW((BoundingBox{lo, {-4, 5, 6}}), celeritas::DebugError);
        EXPECT_THROW((BoundingBox{lo, {4, -5, 6}}), celeritas::DebugError);
        EXPECT_THROW((BoundingBox{lo, {4, 5, -6}}), celeritas::DebugError);
    }

    BoundingBox bb{{-1, -2, 3}, {4, 5, 6}};
    EXPECT_TRUE(bb);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, 3}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, 5, 6}), bb.upper());
}
