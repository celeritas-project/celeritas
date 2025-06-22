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
TEST(EnclosedAziTest, errors)
{
    EXPECT_THROW(EnclosedAzi(Turn{0}, Turn{-0.5}), RuntimeError);
    EXPECT_THROW(EnclosedAzi(Turn{0}, Turn{0}), RuntimeError);
    EXPECT_THROW(EnclosedAzi(Turn{0}, Turn{1.5}), RuntimeError);
}

TEST(EnclosedAziTest, null)
{
    EnclosedAzi azi;
    EXPECT_FALSE(azi);
}

TEST(EnclosedAziTest, transforming)
{
    {
        EnclosedAzi azi{Turn{-2.5}, Turn{-2}};
        EXPECT_REAL_EQ(0.5, azi.start().value());
        EXPECT_REAL_EQ(1.0, azi.stop().value());
    }
    {
        EnclosedAzi azi{Turn{-1.5}, Turn{-0.75}};
        EXPECT_REAL_EQ(0.5, azi.start().value());
        EXPECT_REAL_EQ(1.25, azi.stop().value());
    }
    {
        EnclosedAzi azi{Turn{-0.25}, Turn{-0.125}};
        EXPECT_REAL_EQ(0.75, azi.start().value());
        EXPECT_REAL_EQ(0.875, azi.stop().value());
    }
}

TEST(EnclosedAziTest, make_sense_region)
{
    {
        SCOPED_TRACE("concave");
        EnclosedAzi azi(Turn{-0.25}, Turn{-0.15});
        auto&& [sense, wedge] = azi.make_sense_region();
        EXPECT_EQ(Sense::inside, sense);
        EXPECT_REAL_EQ(0.75, wedge.start().value());
        EXPECT_REAL_EQ(0.85, wedge.stop().value());
    }
    {
        SCOPED_TRACE("convex");
        EnclosedAzi azi(Turn{0.25}, Turn{1.125});
        auto&& [sense, wedge] = azi.make_sense_region();
        EXPECT_EQ(Sense::outside, sense);
        EXPECT_REAL_EQ(1.125, wedge.start().value());
        EXPECT_REAL_EQ(1.25, wedge.stop().value());
    }
    {
        SCOPED_TRACE("half");
        EnclosedAzi azi(Turn{0.1}, Turn{0.6});
        auto&& [sense, wedge] = azi.make_sense_region();
        EXPECT_EQ(Sense::inside, sense);
        EXPECT_REAL_EQ(0.1, wedge.start().value());
        EXPECT_REAL_EQ(0.6, wedge.stop().value());
    }
    {
        SCOPED_TRACE("pac-man");
        EnclosedAzi azi(Turn{0.125}, Turn{0.875});
        auto&& [sense, wedge] = azi.make_sense_region();
        EXPECT_EQ(Sense::outside, sense);
        EXPECT_REAL_EQ(0.875, wedge.start().value());
        EXPECT_REAL_EQ(1.125, wedge.stop().value());
    }
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
    EXPECT_THROW(
        ConeSolid("cone", Cone{{1, 2}, 10.0}, Cone{{1.1, 1.9}, 10.0}, {}),
        RuntimeError);
    // No exclusion, no wedge
    EXPECT_THROW(ConeSolid("cone", Cone{{1, 2}, 10.0}, {}, {}), RuntimeError);
}

TEST_F(SolidTest, inner)
{
    ConeSolid cone("cone", Cone{{1, 2}, 10.0}, Cone{{0.9, 1.9}, 10.0}, {});
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
        "cone@exc.mz,cone@int.mz",
        "cone@exc.pz,cone@int.pz",
        "",
        "cone@int.kz",
        "",
        "cone@int",
        "cone@exc.kz",
        "",
        "cone@exc",
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
    ConeSolid cone(
        "cone", Cone{{1, 2}, 10.0}, {}, EnclosedAzi{Turn{-0.125}, Turn{0.125}});
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
        "cone@int.mz",
        "cone@int.pz",
        "",
        "cone@int.kz",
        "",
        "cone@int",
        "cone@azi.p0",
        "cone@azi.p1",
        "cone@azi",
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
    ConeSolid cone(
        "cone", Cone{{1, 2}, 10.0}, {}, EnclosedAzi{Turn{0.125}, Turn{0.875}});
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
        "cone@int.mz",
        "cone@int.pz",
        "",
        "cone@int.kz",
        "",
        "cone@int",
        "cone@azi.p0",
        "cone@azi.p1",
        "cone@azi",
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
                   EnclosedAzi{Turn{-0.125}, Turn{0.125}});
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
        "cone@exc.mz,cone@int.mz",
        "cone@exc.pz,cone@int.pz",
        "",
        "cone@int.kz",
        "",
        "cone@int",
        "cone@exc.kz",
        "",
        "cone@exc",
        "",
        "cone@azi.p0",
        "cone@azi.p1",
        "cone@azi",
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
    this->build_volume(CylinderSolid("cyl",
                                     Cylinder{1, 10.0},
                                     Cylinder{0.9, 10.0},
                                     EnclosedAzi{Turn{-0.125}, Turn{0.125}}));

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
            "cone", Cone{{1, 2}, 10.0}, std::nullopt, EnclosedAzi{});
        EXPECT_TRUE(shape);
        EXPECT_TRUE(dynamic_cast<ConeShape const*>(shape.get()))
            << "actual shape: " << demangle_shape(*shape);
    }
    {
        auto solid = ConeSolid::or_shape(
            "cone", Cone{{1.1, 2}, 10.0}, Cone{{0.9, 1.9}, 10.0}, EnclosedAzi{});
        EXPECT_TRUE(solid);
        EXPECT_TRUE(dynamic_cast<ConeSolid const*>(solid.get()))
            << demangle_shape(*solid);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
