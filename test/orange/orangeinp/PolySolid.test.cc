//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
using Real2 = PolySegments::Real2;

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
        "pc@0.interior.mz",
        "pc@0.interior.pz,pc@1.interior.mz",
        "",
        "pc@0.interior.kz",
        "",
        "pc",
        "pc@1.interior.pz,pc@2.interior.mz",
        "",
        "pc@1.interior.cz",
        "",
        "pc",
        "pc@2.interior.pz",
        "",
        "pc@2.interior.kz",
        "",
        "pc",
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
        "any(all(all(+0, -1, -2), !all(+0, -1, -3)), all(all(+1, -4, -5), "
        "!all(+1, -4, -6)), all(all(+4, -7, -8), !all(+4, -7, -9)))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.excluded.mz,pc@0.interior.mz",
        "pc@0.excluded.pz,pc@0.interior.pz,pc@1.excluded.mz,pc@1.interior.mz",
        "",
        "pc@0.interior.kz",
        "",
        "pc",
        "pc@0.excluded.cz",
        "",
        "pc",
        "",
        "pc@0",
        "pc@1.excluded.pz,pc@1.interior.pz,pc@2.excluded.mz,pc@2.interior.mz",
        "",
        "pc@1.interior.cz",
        "",
        "pc",
        "pc@1.excluded.kz",
        "",
        "pc",
        "",
        "pc@1",
        "pc@2.excluded.pz,pc@2.interior.pz",
        "",
        "pc@2.interior.kz",
        "",
        "pc",
        "pc@2.excluded.kz",
        "",
        "pc",
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
                                SolidEnclosedAngle{Turn{0.125}, Turn{0.75}}});

    static char const* const expected_surface_strings[] = {
        "Plane: z=-2",
        "Plane: z=0",
        "Cone z: t=0.5 at {0,0,2}",
        "Plane: z=2",
        "Cone z: t=1 at {0,0,-1}",
        "Plane: n={0.70711,0.70711,0}, d=0",
        "Plane: n={0.70711,-0.70711,0}, d=0",
    };
    static char const* const expected_volume_strings[] = {
        "all(any(all(+0, -1, -2), all(+1, -3, -4)), !all(+5, +6))",
    };
    static char const* const expected_md_strings[] = {
        "",
        "",
        "pc@0.interior.mz",
        "pc@0.interior.pz,pc@1.interior.mz",
        "",
        "pc@0.interior.kz",
        "",
        "pc",
        "pc@1.interior.pz",
        "",
        "pc@1.interior.kz",
        "",
        "pc",
        "pc@segments",
        "pc@angle.p0",
        "pc@angle.p1",
        "pc",
        "",
        "pc@restricted",
    };

    auto const& u = this->unit();
    EXPECT_VEC_EQ(expected_surface_strings, surface_strings(u));
    EXPECT_VEC_EQ(expected_volume_strings, volume_strings(u));
    EXPECT_VEC_EQ(expected_md_strings, md_strings(u));
}

TEST_F(PolyconeTest, or_solid)
{
    {
        auto s = PolyCone::or_solid(
            "cone", PolySegments{{1, 2}, {-2, 2}}, SolidEnclosedAngle{});
        EXPECT_TRUE(s);
        EXPECT_TRUE(dynamic_cast<ConeShape const*>(s.get()));
        this->build_volume(*s);
    }
    {
        auto s = PolyCone::or_solid("hollowcone",
                                    PolySegments{{0.5, 0.75}, {1, 2}, {-2, 2}},
                                    SolidEnclosedAngle{});
        EXPECT_TRUE(s);
        EXPECT_TRUE(dynamic_cast<ConeSolid const*>(s.get()));
        this->build_volume(*s);
    }
    {
        auto s = PolyCone::or_solid(
            "transcyl", PolySegments{{2, 2}, {0, 4}}, SolidEnclosedAngle{});
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
        "all(all(+0, -1, -2), !all(+0, -1, -3))",
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
