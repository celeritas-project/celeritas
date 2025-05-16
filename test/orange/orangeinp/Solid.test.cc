//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Solid.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/Solid.hh"

#include "corecel/sys/TypeDemangler.hh"
#include "orange/orangeinp/Shape.hh"

#include "CsgTestUtils.hh"
#include "ObjectTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(SolidEnclosedAngleTest, errors)
{
    EXPECT_THROW(SolidEnclosedAngle(Turn{0}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(SolidEnclosedAngle(Turn{0}, Turn{0}), RuntimeError);
    EXPECT_THROW(SolidEnclosedAngle(Turn{0}, Turn{1.5}), RuntimeError);
}

TEST(SolidEnclosedAngleTest, null)
{
    SolidEnclosedAngle sea;
    EXPECT_FALSE(sea);
}

TEST(SolidEnclosedAngleTest, make_wedge)
{
    {
        SCOPED_TRACE("concave");
        SolidEnclosedAngle sea(Turn{-0.25}, Turn{0.1});
        auto&& [sense, wedge] = sea.make_wedge();
        EXPECT_EQ(Sense::inside, sense);
        EXPECT_SOFT_EQ(0.75, wedge.start().value());
        EXPECT_SOFT_EQ(0.1, wedge.interior().value());
    }
    {
        SCOPED_TRACE("convex");
        SolidEnclosedAngle sea(Turn{0.25}, Turn{0.8});
        auto&& [sense, wedge] = sea.make_wedge();
        EXPECT_EQ(Sense::outside, sense);
        EXPECT_SOFT_EQ(0.05, wedge.start().value());
        EXPECT_SOFT_EQ(0.2, wedge.interior().value());
    }
    {
        SCOPED_TRACE("half");
        SolidEnclosedAngle sea(Turn{0.1}, Turn{0.5});
        auto&& [sense, wedge] = sea.make_wedge();
        EXPECT_EQ(Sense::inside, sense);
        EXPECT_SOFT_EQ(0.1, wedge.start().value());
        EXPECT_SOFT_EQ(0.5, wedge.interior().value());
    }
}

//---------------------------------------------------------------------------//
TEST(SolidZSlabTest, errors)
{
    EXPECT_THROW(SolidZSlab(0.1, -0.1), RuntimeError);
    EXPECT_THROW(SolidZSlab(0.1, 0.1), RuntimeError);
}

TEST(SolidZSlabTest, infinite)
{
    // The default slab spans R^3 and evaluates to false
    SolidZSlab szs;
    EXPECT_FALSE(szs);
}

TEST(SolidZSlabTest, make_wedge)
{
    SolidZSlab szs(5, 10);
    auto inf_slab = szs.make_inf_slab();
    EXPECT_SOFT_EQ(5, inf_slab.lower());
    EXPECT_SOFT_EQ(10, inf_slab.upper());
}

//---------------------------------------------------------------------------//
class SolidTest : public ObjectTestBase
{
  protected:
    Tol tolerance() const override { return Tol::from_relative(1e-4); }
};

//---------------------------------------------------------------------------//
TEST_F(SolidTest, errors)
{
    // Inner region is outside outer
    EXPECT_THROW(ConeSolid("cone", Cone{{1, 2}, 10.0}, Cone{{1.1, 1.9}, 10.0}),
                 RuntimeError);
    // No exclusion, no wedge
    EXPECT_THROW(ConeSolid("cone", Cone{{1, 2}, 10.0}, SolidEnclosedAngle{}),
                 RuntimeError);
    EXPECT_THROW(
        ConeSolid(
            "cone", Cone{{1, 2}, 10.0}, std::nullopt, SolidEnclosedAngle{}),
        RuntimeError);
}

TEST_F(SolidTest, inner)
{
    ConeSolid cone("cone", Cone{{1, 2}, 10.0}, Cone{{0.9, 1.9}, 10.0});
    this->build_volume(cone);

    static char const* const expected_surface_strings[] = {
        "Plane: z=-10",
        "Plane: z=10",
        "Cone z: t=0.05 at {0,0,-30}",
        "Cone z: t=0.05 at {0,0,-28}",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, !all(+0, -1, -3))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cone@excluded.mz,cone@interior.mz",
        "cone@excluded.pz,cone@interior.pz",
        "",
        "cone@interior.kz",
        "",
        "cone@interior",
        "cone@excluded.kz",
        "",
        "cone@excluded",
        "",
        "cone",
    };
    static char const* const expected_bound_strings[] = {
        "7: {{{-0.707,-0.707,-10}, {0.707,0.707,10}}, {{-2,-2,-10}, "
        "{2,2,10}}}",
        "10: {{{-0.672,-0.672,-9}, {0.672,0.672,10}}, {{-1.9,-1.9,-10}, "
        "{1.9,1.9,10}}}",
        "~11: {{{-0.672,-0.672,-9}, {0.672,0.672,10}}, {{-1.9,-1.9,-10}, "
        "{1.9,1.9,10}}}",
        "12: {null, {{-2,-2,-10}, {2,2,10}}}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(SolidTest, wedge)
{
    ConeSolid cone("cone",
                   Cone{{1, 2}, 10.0},
                   SolidEnclosedAngle{Turn{-0.125}, Turn{0.25}});
    this->build_volume(cone);
    static char const* const expected_surface_strings[] = {
        "Plane: z=-10",
        "Plane: z=10",
        "Cone z: t=0.05 at {0,0,-30}",
        "Plane: n={0.70711,0.70711,0}, d=0",
        "Plane: n={0.70711,-0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, +3, +4)",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cone@interior.mz",
        "cone@interior.pz",
        "",
        "cone@interior.kz",
        "",
        "cone@interior",
        "cone@angle.p0",
        "cone@angle.p1",
        "cone@angle",
        "cone",
    };
    // clang-format off
    static char const* const expected_bound_strings[] = {
        "7: {{{-0.707,-0.707,-10}, {0.707,0.707,10}}, {{-2,-2,-10}, {2,2,10}}}",
        "10: {null, inf}",
        "11: {null, {{-2,-2,-10}, {2,2,10}}}",
    };
    // clang-format on

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(SolidTest, antiwedge)
{
    ConeSolid cone("cone",
                   Cone{{1, 2}, 10.0},
                   SolidEnclosedAngle{Turn{0.125}, Turn{0.75}});
    this->build_volume(cone);
    static char const* const expected_surface_strings[] = {
        "Plane: z=-10",
        "Plane: z=10",
        "Cone z: t=0.05 at {0,0,-30}",
        "Plane: n={0.70711,0.70711,0}, d=0",
        "Plane: n={0.70711,-0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, !all(+3, +4))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cone@interior.mz",
        "cone@interior.pz",
        "",
        "cone@interior.kz",
        "",
        "cone@interior",
        "cone@angle.p0",
        "cone@angle.p1",
        "cone@angle",
        "",
        "cone",
    };
    // clang-format off
    static char const* const expected_bound_strings[] = {
        "7: {{{-0.707,-0.707,-10}, {0.707,0.707,10}}, {{-2,-2,-10}, {2,2,10}}}",
        "10: {null, inf}",
        "~11: {null, inf}",
        "12: {null, {{-2,-2,-10}, {2,2,10}}}",
    };
    // clang-format on

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(SolidTest, both)
{
    ConeSolid cone("cone",
                   Cone{{1, 2}, 10.0},
                   Cone{{0.9, 1.9}, 10.0},
                   SolidEnclosedAngle{Turn{-0.125}, Turn{0.25}});
    this->build_volume(cone);
    static char const* const expected_surface_strings[] = {
        "Plane: z=-10",
        "Plane: z=10",
        "Cone z: t=0.05 at {0,0,-30}",
        "Cone z: t=0.05 at {0,0,-28}",
        "Plane: n={0.70711,0.70711,0}, d=0",
        "Plane: n={0.70711,-0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, !all(+0, -1, -3), +4, +5)",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cone@excluded.mz,cone@interior.mz",
        "cone@excluded.pz,cone@interior.pz",
        "",
        "cone@interior.kz",
        "",
        "cone@interior",
        "cone@excluded.kz",
        "",
        "cone@excluded",
        "",
        "cone@angle.p0",
        "cone@angle.p1",
        "cone@angle",
        "cone",
    };
    static char const* const expected_bound_strings[] = {
        "7: {{{-0.707,-0.707,-10}, {0.707,0.707,10}}, {{-2,-2,-10}, "
        "{2,2,10}}}",
        "10: {{{-0.672,-0.672,-9}, {0.672,0.672,10}}, {{-1.9,-1.9,-10}, "
        "{1.9,1.9,10}}}",
        "~11: {{{-0.672,-0.672,-9}, {0.672,0.672,10}}, {{-1.9,-1.9,-10}, "
        "{1.9,1.9,10}}}",
        "14: {null, inf}",
        "15: {null, {{-2,-2,-10}, {2,2,10}}}",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
    EXPECT_VEC_EQ(expected_bound_strings, bound_strings(u));
}

TEST_F(SolidTest, cyl)
{
    this->build_volume(
        CylinderSolid("cyl",
                      Cylinder{1, 10.0},
                      Cylinder{0.9, 10.0},
                      SolidEnclosedAngle{Turn{-0.125}, Turn{0.25}}));

    static char const* const expected_surface_strings[] = {
        "Plane: z=-10",
        "Plane: z=10",
        "Cyl z: r=1",
        "Cyl z: r=0.9",
        "Plane: n={0.70711,0.70711,0}, d=0",
        "Plane: n={0.70711,-0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2, !all(+0, -1, -3), +4, +5)",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
}

TEST_F(SolidTest, or_shape)
{
    TypeDemangler<ObjectInterface> demangle_shape;
    {
        auto shape = ConeSolid::or_shape(
            "cone", Cone{{1, 2}, 10.0}, std::nullopt, SolidEnclosedAngle{});
        EXPECT_TRUE(shape);
        EXPECT_TRUE(dynamic_cast<ConeShape const*>(shape.get()))
            << "actual shape: " << demangle_shape(*shape);
    }
    {
        auto solid = ConeSolid::or_shape("cone",
                                         Cone{{1.1, 2}, 10.0},
                                         Cone{{0.9, 1.9}, 10.0},
                                         SolidEnclosedAngle{});
        EXPECT_TRUE(solid);
        EXPECT_TRUE(dynamic_cast<ConeSolid const*>(solid.get()))
            << demangle_shape(*solid);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
