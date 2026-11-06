//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.test.cc
//---------------------------------------------------------------------------//

#include "orange/surf/Tri.hh"

#include "orange/OrangeTypes.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

class TriTest : public ::celeritas::test::Test
{
  protected:
    using real_type = float;
    using Intersections = Tri<real_type>::Intersections;
    using Real3 = Tri<real_type>::Real3;
};

TEST_F(TriTest, construction)
{
    Real3 a = {0.0, 0.0, 0.0};
    Real3 b = {1.0, 0, 0};
    Real3 c = {0.5, 1.0, 1.0};
    Tri<real_type> tri({a, b, c});

    Real3 a_gold = {0, 0, 0};
    Real3 b_gold = {1, 0, 0};
    Real3 c_gold = {0.5, 1, 1};
    EXPECT_VEC_SOFT_EQ(a_gold, tri.verts()[0]);
    EXPECT_VEC_SOFT_EQ(b_gold, tri.verts()[1]);
    EXPECT_VEC_SOFT_EQ(c_gold, tri.verts()[2]);
}

TEST_F(TriTest, calc_intersections)
{
    Real3 a = {0.0, 0.0, 0.0};
    Real3 b = {1.0, 0, 0};
    Real3 c = {0.5, 1.0, 1.0};
    Tri<real_type> tri({a, b, c});

    Real3 pos = {0.4f, 0.4f, 0};
    Real3 dir = {0, 0, 1};

    auto t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(0.4f, t[0]);

    // Starting on the intersection point pointing away yields no intersection
    pos = {0.4f, 0.4f, real_type(t[0])};
    t = tri.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(real_type(no_intersection()), t[0]);

    // Ray is parallel to the tri
    pos = {0.4f, 0.4f, 0.2f};
    dir = {0.0f, -0.7071067811865475f, -0.7071067811865475f};
    t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(real_type(no_intersection()), t[0]);

    // No intersection with the triangle
    pos = {0.4f, 0.4f, 0.2f};
    dir = {0.0, -0.3846153846153846, -0.923076923076923};
    t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(real_type(no_intersection()), t[0]);

    // Ray intersects edge 2 outside of the triangle
    pos = {0.4f, 0.4f, 0.0f};
    dir = {-0.8320502943378437f, -0.5547001962252291f, 0.0};
    t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(real_type(no_intersection()), t[0]);

    // Ray intersects edge 1 outside of triangle
    pos = {0.4f, -0.4f, 0.0f};
    dir = {0, -1, 0};
    t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(real_type(no_intersection()), t[0]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
