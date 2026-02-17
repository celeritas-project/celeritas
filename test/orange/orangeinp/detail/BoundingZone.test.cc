//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
std::string to_string(BoundingZone const& bz)
{
    std::ostringstream os;
    os << bz;
    return std::move(os).str();
}

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

        EXPECT_TRUE(bz) << "Invalid bz: " << bz;

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

TEST_F(BoundingZoneTest, degenerate)
{
    Real3 const wherever{0.9, 0.9, 0};
    BoundingZone e;
    EXPECT_EQ(IsInside::no, is_inside(e, wherever));
    EXPECT_EQ("{nowhere}", to_string(e));
    e.negate();
    EXPECT_EQ(IsInside::yes, is_inside(e, wherever));
    EXPECT_EQ("{everywhere}", to_string(e));

    e = BoundingZone::from_infinite();
    EXPECT_EQ(IsInside::yes, is_inside(e, wherever));
    EXPECT_EQ("{everywhere}", to_string(e));
    e.negate();
    EXPECT_EQ(IsInside::no, is_inside(e, wherever));
    EXPECT_EQ("{nowhere}", to_string(e));

    // Indefinite
    e = BoundingZone::from_infinite();
    e.interior = {};
    EXPECT_EQ(IsInside::maybe, is_inside(e, wherever));
    EXPECT_EQ("{maybe anywhere}", to_string(e));
    e.negate();
    EXPECT_EQ(IsInside::maybe, is_inside(e, wherever));
    EXPECT_EQ("{maybe anywhere}", to_string(e));
}

TEST_F(BoundingZoneTest, standard)
{
    auto sph = make_bz({0, 0, 0}, 1.0, 0.7);
    EXPECT_EQ(IsInside::no, is_inside(sph, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(sph, {0.9, 0.9, 0}));
    EXPECT_EQ(IsInside::yes, is_inside(sph, {0.5, 0.5, 0.5}));
    EXPECT_EQ(
        R"({always inside {{-0.7,-0.7,-0.7}, {0.7,0.7,0.7}} and never outside {{-1,-1,-1}, {1,1,1}}})",
        to_string(sph));

    // Invert
    sph.negate();
    EXPECT_EQ(IsInside::yes, is_inside(sph, {1.01, 0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(sph, {0.9, 0.9, 0}));
    EXPECT_EQ(IsInside::no, is_inside(sph, {0.5, 0.5, 0.5}));
    EXPECT_EQ(
        R"({never inside {{-0.7,-0.7,-0.7}, {0.7,0.7,0.7}} and always outside {{-1,-1,-1}, {1,1,1}}})",
        to_string(sph));

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
    EXPECT_EQ("{never outside {{-1.5,-1.5,-1.5}, {1.5,1.5,1.5}}}",
              to_string(extonly));

    // Invert
    extonly.negate();
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {0.0, 0.0, 0}));
    EXPECT_EQ(IsInside::maybe, is_inside(extonly, {1.4, 0, 0}));
    EXPECT_EQ(IsInside::yes, is_inside(extonly, {2.0, 0, 0}));
    EXPECT_EQ("{always outside {{-1.5,-1.5,-1.5}, {1.5,1.5,1.5}}}",
              to_string(extonly));
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
        EXPECT_VEC_SOFT_EQ((Real3{-1, -1, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1, 1, 1}), bz.exterior.upper());
    }
    {
        auto box = make_bz({0.5, 0, 0}, 0.5, 0.5);
        auto large = make_bz({0, 0, 0}, 1.0, 1.0);
        auto bz = calc_intersection(box, large);
        EXPECT_EQ(box.interior, bz.interior);
        EXPECT_EQ(box.exterior, bz.exterior);
        EXPECT_FALSE(bz.negated);

        bz = calc_intersection(box, box);
        EXPECT_EQ(box.interior, bz.interior);
        EXPECT_EQ(box.exterior, bz.exterior);
        EXPECT_FALSE(bz.negated);
    }
    {
        // Degenerate test: edges are "in"
        auto box = make_bz({0.5, 0, 0}, 0.5, 0.5);
        auto negbox = negated_bz(box);
        auto bz = calc_intersection(box, negbox);
        EXPECT_EQ(box.interior, bz.interior);
        EXPECT_EQ(box.exterior, bz.exterior);
        EXPECT_FALSE(bz.negated);
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
        EXPECT_VEC_SOFT_EQ((Real3{0.5, -0.5, -0.5}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{1.5, 0.5, 0.5}), bz.exterior.upper());
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
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, -1}), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ((Real3{2, 2, 1}), bz.exterior.upper());
    }
    {
        // Union with null should be the same as non-null
        auto bz = calc_union(BoundingZone{}, sph);
        EXPECT_FALSE(bz.negated);
        EXPECT_VEC_SOFT_EQ(sph.interior.lower(), bz.interior.lower());
        EXPECT_VEC_SOFT_EQ(sph.interior.upper(), bz.interior.upper());
        EXPECT_VEC_SOFT_EQ(sph.exterior.lower(), bz.exterior.lower());
        EXPECT_VEC_SOFT_EQ(sph.exterior.upper(), bz.exterior.upper());
    }
}

/*!
 * Test an intersection of unions.
 *
 * Unsimplified volume, node 62:
   all(+12, -13, +14, -15, +16, -17,
    ~all(+18, -19, +20, -21, +22, -23),
    ~all(+18, -19, +20, -21, +24, -25),
    ~all(+18, -19, +20, -21, +26, -27),
    ~all(+18, -19, +20, -21, +28, -29)
   )
 *
 * i.e.,
  = &[ 32,43,49,55,61]
 32: {{{-1.15,-618,-560}, {1.15,-606,-350}},
      {{-1.15,-618,-560}, {1.15,-606,-350}}}
~43: {{{-1.2,-617,-559}, {1.2,-608,-512}},
      {{-1.2,-617,-559}, {1.2,-608,-512}}}
~49: {{{-1.2,-617,-510}, {1.2,-608,-463}},
      {{-1.2,-617,-510}, {1.2,-608,-463}}}
~55: {{{-1.2,-617,-447}, {1.2,-608,-400}},
      {{-1.2,-617,-447}, {1.2,-608,-400}}}
~61: {{{-1.2,-617,-398}, {1.2,-608,-351}},
      {{-1.2,-617,-398}, {1.2,-608,-351}}}
 *
 * See g4org/ProtoConstructor.test.cc::DuneCryostatTest
 */
TEST_F(BoundingZoneTest, arapuca_walls)
{
    // Outer
    BoundingZone bz;
    bz.interior = {{-1.15, -618, -560}, {1.15, -606, -350}};
    bz.exterior = {{-1.15, -618, -560}, {1.15, -606, -350}};
    auto subtract
        = [&bz](BoundingBox<> const& inner, BoundingBox<> const& outer) {
              BoundingZone rhs{inner, outer};
              rhs.negate();
              bz = calc_intersection(bz, rhs);
          };

    subtract({{-1.2, -617, -559}, {1.2, -608, -512}},
             {{-1.2, -617, -559}, {1.2, -608, -512}});
    subtract({{-1.2, -617, -510}, {1.2, -608, -463}},
             {{-1.2, -617, -510}, {1.2, -608, -463}});
    subtract({{-1.2, -617, -447}, {1.2, -608, -400}},
             {{-1.2, -617, -447}, {1.2, -608, -400}});
    subtract({{-1.2, -617, -398}, {1.2, -608, -351}},
             {{-1.2, -617, -398}, {1.2, -608, -351}});

    EXPECT_FALSE(bz.negated);
    EXPECT_FALSE(bz.interior) << bz.interior;
    EXPECT_VEC_SOFT_EQ((Real3{-1.15, -618, -560}), bz.exterior.lower());
    EXPECT_VEC_SOFT_EQ((Real3{1.15, -606, -350}), bz.exterior.upper());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
