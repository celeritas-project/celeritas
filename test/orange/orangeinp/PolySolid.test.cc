//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/PolySolid.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/PolySolid.hh"

#include "orange/orangeinp/Shape.hh"
#include "orange/orangeinp/Solid.hh"
#include "orange/orangeinp/Transformed.hh"

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
TEST(PolySegmentsTest, errors)
{
    // Not enough elements
    EXPECT_THROW(PolySegments({}, {}), RuntimeError);
    EXPECT_THROW(PolySegments({1}, {2}), RuntimeError);
    // Inconsistent sizes
    EXPECT_THROW(PolySegments({1}, {2, 2}), RuntimeError);
    // Out of order Z
    EXPECT_THROW(PolySegments({1, 2, 3}, {2, 1, 3}), RuntimeError);
    // Invalid inner size
    EXPECT_THROW(PolySegments({1, 2}, {2, 2}, {3, 4, 5}), RuntimeError);
    // Inner outside outer
    EXPECT_THROW(PolySegments({3, 3}, {2, 3}, {0, 1}), RuntimeError);
}

TEST(PolySegmentsTest, filled)
{
    PolySegments seg({2, 1, 3, 4}, {-1, 0, 2, 6});
    EXPECT_EQ(3, seg.size());
    EXPECT_FALSE(seg.has_exclusion());
    EXPECT_VEC_EQ((Real2{2, 1}), seg.outer(0));
    EXPECT_VEC_EQ((Real2{1, 3}), seg.outer(1));
    EXPECT_VEC_EQ((Real2{3, 4}), seg.outer(2));
    EXPECT_VEC_EQ((Real2{-1, 0}), seg.z(0));
    EXPECT_VEC_EQ((Real2{2, 6}), seg.z(2));
}

TEST(PolySegmentsTest, hollow)
{
    PolySegments seg({1, 0.5, 2.5, 2}, {2, 1, 3, 4}, {-1, 0, 2, 6});
    EXPECT_EQ(3, seg.size());
    EXPECT_TRUE(seg.has_exclusion());
    EXPECT_VEC_EQ((Real2{1, 0.5}), seg.inner(0));
    EXPECT_VEC_EQ((Real2{2.5, 2}), seg.inner(2));
    EXPECT_VEC_EQ((Real2{2, 1}), seg.outer(0));
    EXPECT_VEC_EQ((Real2{3, 4}), seg.outer(2));
    EXPECT_VEC_EQ((Real2{-1, 0}), seg.z(0));
    EXPECT_VEC_EQ((Real2{2, 6}), seg.z(2));

    // Reversed order
    using VecReal = PolySegments::VecReal;
    PolySegments rev({2, 1, 4}, {3, 2, 5}, {6, 4, 1});
    EXPECT_VEC_EQ(VecReal({4, 1, 2}), rev.inner());
    EXPECT_VEC_EQ(VecReal({5, 2, 3}), rev.outer());
    EXPECT_VEC_EQ(VecReal({1, 4, 6}), rev.z());
}

//---------------------------------------------------------------------------//
class PolyconeTest : public ObjectTestBase
{
  protected:
    Tol tolerance() const override { return Tol::from_relative(1e-4); }
};

TEST_F(PolyconeTest, filled)
{
    this->build_volume(
        PolyCone{"pc", PolySegments{{2, 1, 1, 3}, {-2, -1, 0, 2}}, {}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=-1",
        "Cone z: t=1 at {0,0,0}",
        "Plane: z=0",
        "Cyl z: r=1",
        "Plane: z=2",
        "Cone z: t=1 at {0,0,-1}",
    };
    static char const* const expected_volume_strings[] = {
        "any(all(+0, -1, -2), all(+1, -3, -4), all(+3, -5, -6))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.int.mz",
        "pc@0.int.pz,pc@1.int.mz",
        "",
        "pc@0.int.kz",
        "",
        "pc@0.int",
        "pc@1.int.pz,pc@2.int.mz",
        "",
        "pc@1.int.cz",
        "",
        "pc@1.int",
        "pc@2.int.pz",
        "",
        "pc@2.int.kz",
        "",
        "pc@2.int",
        "pc@segments",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(PolyconeTest, hollow)
{
    this->build_volume(PolyCone{
        "pc",
        PolySegments{{0.5, 0.5, 0.75, 1}, {2, 1, 1, 3}, {-2, -1, 0, 2}},
        {}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=-1",
        "Cone z: t=1 at {0,0,0}",
        "Cyl z: r=0.5",
        "Plane: z=0",
        "Cyl z: r=1",
        "Cone z: t=0.25 at {0,0,-3}",
        "Plane: z=2",
        "Cone z: t=1 at {0,0,-1}",
        "Cone z: t=0.125 at {0,0,-6}",
    };
    static char const* const expected_volume_strings[] = {
        R"(any(all(+0, -1, -2, !all(+0, -1, -3)), all(+1, -4, -5, !all(+1, -4, -6)), all(+4, -7, -8, !all(+4, -7, -9))))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.exc.mz,pc@0.int.mz",
        "pc@0.exc.pz,pc@0.int.pz,pc@1.exc.mz,pc@1.int.mz",
        "",
        "pc@0.int.kz",
        "",
        "pc@0.int",
        "pc@0.exc.cz",
        "",
        "pc@0.exc",
        "",
        "pc@0",
        "pc@1.exc.pz,pc@1.int.pz,pc@2.exc.mz,pc@2.int.mz",
        "",
        "pc@1.int.cz",
        "",
        "pc@1.int",
        "pc@1.exc.kz",
        "",
        "pc@1.exc",
        "",
        "pc@1",
        "pc@2.exc.pz,pc@2.int.pz",
        "",
        "pc@2.int.kz",
        "",
        "pc@2.int",
        "pc@2.exc.kz",
        "",
        "pc@2.exc",
        "",
        "pc@2",
        "pc@segments",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(PolyconeTest, sliced)
{
    this->build_volume(PolyCone{"pc",
                                PolySegments{{2, 1, 3}, {-2, 0, 2}},
                                EnclosedAzi{Turn{0.125}, Turn{0.875}}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=0",
        "Cone z: t=0.5 at {0,0,2}",
        "Plane: z=2",
        "Cone z: t=1 at {0,0,-1}",
        "Plane: n={0.70711,-0.70711,0}, d=0",
        "Plane: n={0.70711,0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(any(all(+0, -1, -2), all(+1, -3, -4)), !all(+5, +6))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.int.mz",
        "pc@0.int.pz,pc@1.int.mz",
        "",
        "pc@0.int.kz",
        "",
        "pc@0.int",
        "pc@1.int.pz",
        "",
        "pc@1.int.kz",
        "",
        "pc@1.int",
        "pc@segments",
        "pc@awm",
        "pc@awp",
        "pc@~azi",
        "",
        "pc@restricted",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(PolyconeTest, degenerate)
{
    this->build_volume(
        PolyCone{"cyls", PolySegments{{2, 2, 1, 1}, {-2, -1, -1, 2}}, {}});
    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=-1",
        "Cyl z: r=2",
        "Plane: z=2",
        "Cyl z: r=1",
    };
    static char const* const expected_volume_strings[] = {
        "any(all(+0, -1, -2), all(+1, -3, -4))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "cyls@0.int.mz",
        "cyls@0.int.pz,cyls@2.int.mz",
        "",
        "cyls@0.int.cz",
        "",
        "cyls@0.int",
        "cyls@2.int.pz",
        "",
        "cyls@2.int.cz",
        "",
        "cyls@2.int",
        "cyls@segments",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

// Try to build a polycone with segments that have zero-width cylinders
// (TGeoPcon0x16 from alice-pipe-its.gdml)
TEST_F(PolyconeTest, degenerate_inner)
{
    this->build_volume(PolyCone{
        "pc",
        PolySegments{
            {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.6, 0.0, 0.0, 0.0, 0.0},
            {3.5, 3.5, 1, 1, 1.75, 1.75, 1, 1, 2.5, 2.5, 1.75, 1.75, 1.75, 0.75, 0.75},
            {0.00,
             1.27,
             1.27,
             4.07,
             4.07,
             5.67,
             5.67,
             10.87,
             10.87,
             12.87,
             12.87,
             18.57,
             18.77,
             18.77,
             19.27}},
        {}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=0",
        "Plane: z=1.27",
        "Cyl z: r=3.5",
        "Cyl z: r=0.8",
        "Plane: z=4.07",
        "Cyl z: r=1",
        "Plane: z=5.67",
        "Cyl z: r=1.75",
        "Plane: z=10.87",
        "Plane: z=12.87",
        "Cyl z: r=2.5",
        "Plane: z=18.57",
        "Cone z: t=0.28070 at {0,0,18.57}",
        "Plane: z=18.77",
        "Plane: z=19.27",
        "Cyl z: r=0.75",
    };
    static char const* const expected_volume_strings[] = {
        R"(any(all(+0, -1, -2, !all(+0, -1, -3)), all(+1, -5, -6, !all(+1, -3, -5)), all(+5, -7, -8, !all(-3, +5, -7)), all(-6, +7, -9, !all(-3, +7, -9)), all(+9, -10, -11, !all(-3, +9, -10)), all(-8, +10, -13, !all(+10, -13, -14)), all(-8, +13, -15), all(+15, -17, -18)))"};
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.exc.mz,pc@0.int.mz",
        "pc@0.exc.pz,pc@0.int.pz,pc@2.exc.mz,pc@2.int.mz",
        "",
        "pc@0.int.cz",
        "",
        "pc@0.int",
        "pc@0.exc.cz,pc@2.exc.cz,pc@4.exc.cz,pc@6.exc.cz,pc@8.exc.cz",
        "",
        "pc@0.exc",
        "",
        "pc@0",
        "pc@2.exc.pz,pc@2.int.pz,pc@4.exc.mz,pc@4.int.mz",
        "",
        "pc@2.int.cz,pc@6.int.cz",
        "",
        "pc@2.int",
        "pc@2.exc",
        "",
        "pc@2",
        "pc@4.exc.pz,pc@4.int.pz,pc@6.exc.mz,pc@6.int.mz",
        "",
        "pc@10.int.cz,pc@11.int.cz,pc@4.int.cz",
        "",
        "pc@4.int",
        "pc@4.exc",
        "",
        "pc@4",
        "pc@6.exc.pz,pc@6.int.pz,pc@8.exc.mz,pc@8.int.mz",
        "",
        "pc@6.int",
        "pc@6.exc",
        "",
        "pc@6",
        "pc@10.exc.mz,pc@10.int.mz,pc@8.exc.pz,pc@8.int.pz",
        "",
        "pc@8.int.cz",
        "",
        "pc@8.int",
        "pc@8.exc",
        "",
        "pc@8",
        "pc@10.exc.pz,pc@10.int.pz,pc@11.int.mz",
        "",
        "pc@10.int",
        "pc@10.exc.kz",
        "",
        "pc@10.exc",
        "",
        "pc@10",
        "pc@11.int.pz,pc@13.int.mz",
        "",
        "pc@11.int",
        "pc@13.int.pz",
        "",
        "pc@13.int.cz",
        "",
        "pc@13.int",
        "pc@segments",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    if constexpr (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_FLOAT)
    {
        // Floating point precision is slightly off in the printout but
        // otherwise correct; the volume strings are different because some of
        // the planes show up as "exactly equal" (deleted) versus "nearly
        // equal" (chained and replaced)
        EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    }

    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(PolyconeTest, or_solid)
{
    {
        auto s = PolyCone::or_solid(
            "cone", PolySegments{{1, 2}, {-2, 2}}, EnclosedAzi{});
        EXPECT_TRUE(s);
        EXPECT_TRUE(dynamic_cast<ConeShape const*>(s.get()));
        this->build_volume(*s);
    }
    {
        auto s = PolyCone::or_solid("hollowcone",
                                    PolySegments{{0.5, 0.75}, {1, 2}, {-2, 2}},
                                    EnclosedAzi{});
        EXPECT_TRUE(s);
        EXPECT_TRUE(dynamic_cast<ConeSolid const*>(s.get()));
        this->build_volume(*s);
    }
    {
        auto s = PolyCone::or_solid(
            "transcyl", PolySegments{{2, 2}, {0, 4}}, EnclosedAzi{});
        EXPECT_TRUE(s);
        EXPECT_TRUE(dynamic_cast<Transformed const*>(s.get()));
        this->build_volume(*s);
    }

    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=2",
        "Cone z: t=0.25 at {0,0,-6}",
        "Cone z: t=0.0625 at {0,0,-10}",
        "Plane: z=0",
        "Plane: z=4",
        "Cyl z: r=2",
    };
    static char const* const expected_volume_strings[] = {
        "all(+0, -1, -2)",
        "all(+0, -1, -2, !all(+0, -1, -3))",
        "all(+4, -5, -6)",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace orangeinp
}  // namespace celeritas
