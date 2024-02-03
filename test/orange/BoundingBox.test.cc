//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/BoundingBox.test.cc
//---------------------------------------------------------------------------//
#include "geocel/BoundingBox.hh"

#include <limits>

#include "celeritas_config.h"
#include "corecel/cont/ArrayIO.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_JSON
#    include "geocel/BoundingBoxIO.json.hh"
#endif

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
    ibb.shrink(Bound::lo, Axis::x, 2);
    ibb.shrink(Bound::hi, Axis::x, 1);
    EXPECT_FALSE(ibb);
}

TEST_F(BoundingBoxTest, degenerate)
{
    // Two coincident square faces
    BBox bbox{{-2, 1, -2}, {2, 1, 2}};
    EXPECT_TRUE(bbox);
    EXPECT_LT(bbox.lower()[0], bbox.upper()[0]);
    EXPECT_EQ(bbox.lower()[1], bbox.upper()[1]);

    BBox ibb = BBox::from_infinite();
    ibb.shrink(Bound::lo, Axis::z, 1);
    ibb.shrink(Bound::hi, Axis::z, 1);
    EXPECT_TRUE(ibb);

    // Triple-degenerate: only contains a single point
    EXPECT_TRUE((BBox{{1, 1, 1}, {1, 1, 1}}));
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
        Real3 const lo{-1, -2, -3};
        EXPECT_THROW((BBox{lo, {-4, 5, 6}}), DebugError);
        EXPECT_THROW((BBox{lo, {4, -5, 6}}), DebugError);
        EXPECT_THROW((BBox{lo, {4, 5, -6}}), DebugError);
    }

    BBox bb{{-1, -2, 3}, {4, 5, 6}};
    EXPECT_TRUE(bb);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, 3}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{4, 5, 6}), bb.upper());

    bb.shrink(Bound::hi, Axis::x, 2);
    bb.shrink(Bound::lo, Axis::z, 4);
    bb.shrink(Bound::hi, Axis::y, 0);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -2, 4}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{2, 0, 6}), bb.upper());

    bb.grow(Bound::hi, Axis::x, 3);
    bb.grow(Bound::lo, Axis::y, -3);
    bb.grow(Bound::hi, Axis::y, -1);  // null op
    bb.grow(Bound::lo, Axis::z, 2);
    EXPECT_VEC_SOFT_EQ((Real3{-1, -3, 2}), bb.lower());
    EXPECT_VEC_SOFT_EQ((Real3{3, 0, 6}), bb.upper());
}

TEST_F(BoundingBoxTest, is_inside)
{
    BBox bbox1 = {{-5, -2, -100}, {6, 1, 1}};
    EXPECT_TRUE(is_inside(bbox1, Real3{-4, 0, 0}));
    EXPECT_TRUE(is_inside(bbox1, Real3{-4.9, -1.9, -99.9}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-6, 0, 0}));
    EXPECT_FALSE(is_inside(bbox1, Real3{-5.1, -2.1, -101.1}));
    EXPECT_FALSE(is_inside(BBox{}, Real3{0, 0, 0}));

    BBox degenerate{{1, -2, -2}, {1, 2, 2}};
    EXPECT_TRUE(is_inside(degenerate, Real3{1, 0, 0}));
    EXPECT_FALSE(is_inside(degenerate, Real3{1, -3, 0}));
    EXPECT_FALSE(is_inside(degenerate, Real3{1.000001, 0, 0}));

    BBox super_degenerate{{1, 1, 1}, {1, 1, 1}};
    EXPECT_TRUE(is_inside(degenerate, Real3{1, 1, 1}));
}

TEST_F(BoundingBoxTest, TEST_IF_CELERITAS_JSON(io))
{
    using BoundingBoxT = BoundingBox<double>;
    auto to_json_string = [](BoundingBoxT const& bbox) {
#if CELERITAS_USE_JSON
        nlohmann::json j = bbox;
        return j.dump();
#else
        return std::string{};
#endif
    };
    auto from_json_string = [](std::string const& s) {
#if CELERITAS_USE_JSON
        return nlohmann::json::parse(s).get<BoundingBoxT>();
#else
        return BoundingBoxT{};
#endif
    };

    static BoundingBoxT const bboxes[] = {
        {{-1, -2, -3}, {3, 2, 1}},
        {{-inf, -2, -3}, {3, 2, inf}},
        BoundingBoxT::from_infinite(),
        BoundingBoxT{},
    };

    EXPECT_EQ("[[-1.0,-2.0,-3.0],[3.0,2.0,1.0]]", to_json_string(bboxes[0]));
    EXPECT_EQ("null", to_json_string(BoundingBoxT{}));

    // Test round tripping
    for (BoundingBoxT const& bb : bboxes)
    {
        BoundingBoxT reconstructed;
        EXPECT_NO_THROW(reconstructed = from_json_string(to_json_string(bb)));
        EXPECT_VEC_SOFT_EQ(bb.lower(), reconstructed.lower());
        EXPECT_VEC_SOFT_EQ(bb.upper(), reconstructed.upper());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
