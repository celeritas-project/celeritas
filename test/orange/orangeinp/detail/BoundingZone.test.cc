//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/BoundingZone.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/BoundingZone.hh"

#include "corecel/io/Repr.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/BoundingBoxUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

class BoundingZoneTest : public ::celeritas::test::Test
{
  protected:
    enum class IsInside
    {
        no = -1,
        maybe = 0,
        yes = 1
    };

  protected:
    static BoundingZone
    make_bz(Real3 center, real_type outer_hw, real_type inner_hw = -1)
    {
        CELER_EXPECT(outer_hw >= 0);
        CELER_EXPECT(outer_hw >= inner_hw);

        BoundingZone result;
        if (inner_hw > 0)
        {
            result.interior = BBox{{center[0] - inner_hw,
                                    center[1] - inner_hw,
                                    center[2] - inner_hw},
                                   {center[0] + inner_hw,
                                    center[1] + inner_hw,
                                    center[2] + inner_hw}};
        }
        result.exterior = BBox{
            {center[0] - outer_hw, center[1] - outer_hw, center[2] - outer_hw},
            {center[0] + outer_hw, center[1] + outer_hw, center[2] + outer_hw}};
        return result;
    }

    static BoundingZone negated_bz(BoundingZone const& bz)
    {
        BoundingZone result{bz};
        result.negate();
        return result;
    }

    // Note that the 'maybe' testing for boundaries is more strict than we need
    // in practicality since we will bump them in practice.
    static IsInside is_inside(BoundingZone const& bz, Real3 const& point)
    {
        using celeritas::is_inside;

        EXPECT_TRUE(encloses(bz.exterior, bz.interior))
            << "Exterior " << bz.exterior << " does not enclose interior "
            << bz.interior;

        if (!is_inside(bz.exterior, point))
        {
            // Strictly outside exterior box
            return bz.negated ? IsInside::yes : IsInside::no;
        }
        if (!is_inside(bz.interior, point))
        {
            // Strictly outside interior box
            return IsInside::maybe;
        }
        // Inside or on interior box
        return bz.negated ? IsInside::no : IsInside::yes;
    }

    static void print_expected(BoundingZone const& bz)
    {
        cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
             << (bz.negated ? "EXPECT_TRUE" : "EXPECT_FALSE")
             << "(bz.negated);\n";

#define BZ_EXPECTED_PT(BOX, POINT)                                          \
    cout << "EXPECT_VEC_SOFT_EQ((Real3" << repr(bz.BOX.POINT()) << "), bz." \
         << #BOX "." #POINT "());\n"
#define BZ_EXPECTED(BOX)                                            \
    if (!bz.BOX)                                                    \
    {                                                               \
        cout << "EXPECT_FALSE(bz." #BOX ") << bz." #BOX ";\n";      \
    }                                                               \
    else if (bz.BOX == BBox::from_infinite())                       \
    {                                                               \
        cout << "EXPECT_EQ(BBox::from_infinite(), bz." #BOX ");\n"; \
    }                                                               \
    else                                                            \
    {                                                               \
        BZ_EXPECTED_PT(BOX, lower);                                 \
        BZ_EXPECTED_PT(BOX, upper);                                 \
    }
        BZ_EXPECTED(interior);
        BZ_EXPECTED(exterior);
#undef BZ_EXPECTED_PT
#undef BZ_EXPECTED
        cout << "/*** END CODE ***/\n";
    }
};

TEST_F(BoundingZoneTest, standard)
{
    auto sph = make_bz({0, 0, 0}, 1.0, 0.7);
    EXPECT_EQ(IsInside::no, is_inside(sph, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(sph, {0.9, 0.9, 0}));
    EXPECT_EQ(IsInside::yes, is_inside(sph, {0.5, 0.5, 0.5}));

    // Invert
    sph.negate();
    EXPECT_EQ(IsInside::yes, is_inside(sph, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(sph, {0.9, 0.9, 0}));
    EXPECT_EQ(IsInside::no, is_inside(sph, {0.5, 0.5, 0.5}));

    auto box = make_bz({0, 0, 0}, 1.0, 1.0);
    EXPECT_EQ(IsInside::no, is_inside(box, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::yes, is_inside(box, {0.9, 0.5, 0.5}));

    box.negate();
    EXPECT_EQ(IsInside::yes, is_inside(box, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::no, is_inside(box, {0.9, 0.5, 0.5}));
}

TEST_F(BoundingZoneTest, exterior_only)
{
    auto extonly = make_bz({0, 0, 0}, 1.5);
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {0.0, 0.0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {1.4, 0, 0}));
    EXPECT_EQ(IsInside::no, is_inside(extonly, {2.0, 0, 0}));

    // Invert
    extonly.negate();
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {0.0, 0.0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {1.4, 0, 0}));
    EXPECT_EQ(IsInside::yes, is_inside(extonly, {2.0, 0, 0}));
}

TEST_F(BoundingZoneTest, calc_intersection)
{
    auto const sph = make_bz({0, 0, 0}, 1.0, 0.7);
    auto const negsph = negated_bz(sph);
    auto const extonly = make_bz({1, 0, 0}, 0.5);
    auto const negextonly = negated_bz(extonly);

    {
        // Outer overlaps inner region along x, is equal to inner on y, extends
        // beyond outer on z
        auto const ovoid = [] {
            BoundingZone result;
            result.exterior = {{0.0, -0.7, -2.0}, {2.0, 0.7, 2.0}};
            result.interior = {{0.1, -0.3, -1.0}, {1.9, 0.3, 1.0}};
            return result;
        }();
        auto bz = calc_intersection(sph, ovoid);
        EXPECT_FALSE(bz.negated);
        EXPECT_VEC_SOFT_EQ((Real3{0, -0.7, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 0.7, 1}), bz.exterior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{0.1, -0.3, -0.7}), bz.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{0.7, 0.3, 0.7}), bz.interior.upper());
    }
    {
        auto bz = calc_intersection(sph, extonly);
        EXPECT_FALSE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_VEC_SOFT_EQ((Real3{0.5, -0.5, -0.5}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 0.5, 0.5}), bz.exterior.upper());
    }
    {
        auto bz = calc_intersection(sph, negextonly);
        EXPECT_FALSE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), bz.exterior.upper());
    }
    {
        auto bz = calc_intersection(negsph, negextonly);
        EXPECT_TRUE(bz.negated);
        EXPECT_VEC_SOFT_EQ((Real3{-0.7, -0.7, -0.7}), bz.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{0.7, 0.7, 0.7}), bz.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1.5, 1, 1}), bz.exterior.upper());
    }
    {
        auto const trasq = make_bz({1.0, 1.0, 0}, 1.0, 0.7);
        auto bz = calc_intersection(sph, negated_bz(trasq));
        EXPECT_FALSE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_EQ(BBox::from_infinite(), bz.exterior);
    }
}

TEST_F(BoundingZoneTest, calc_union)
{
    auto const sph = make_bz({0, 0, 0}, 1.0, 0.7);
    auto const trasph = make_bz({1.0, 1.0, 0}, 1.0, 0.7);
    auto const extonly = make_bz({1, 0, 0}, 0.5);
    auto const negextonly = negated_bz(extonly);

    {
        auto bz = calc_union(sph, trasph);
        EXPECT_FALSE(bz.negated);
        EXPECT_VEC_SOFT_EQ((Real3{0.3, 0.3, -0.7}), bz.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1.7, 1.7, 0.7}), bz.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1}), bz.exterior.upper());
    }
    {
        auto bz = calc_union(sph, extonly);
        EXPECT_FALSE(bz.negated);
        EXPECT_VEC_SOFT_EQ((Real3{-0.7, -0.7, -0.7}), bz.interior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{0.7, 0.7, 0.7}), bz.interior.upper());
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1.5, 1, 1}), bz.exterior.upper());
    }
    {
        auto bz = calc_union(sph, negextonly);
        EXPECT_TRUE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), bz.exterior.upper());
    }
    {
        auto bz = calc_union(negated_bz(sph), negextonly);
        EXPECT_TRUE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_VEC_SOFT_EQ((Real3{0.5, -0.5, -0.5}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 0.5, 0.5}), bz.exterior.upper());
    }
    {
        auto bz = calc_union(sph, negated_bz(trasph));
        EXPECT_TRUE(bz.negated);
        EXPECT_FALSE(bz.interior) << bz.interior;
        EXPECT_EQ(BBox::from_infinite(), bz.exterior);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
