//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/Tri.test.cc
//---------------------------------------------------------------------------//

#include "orange/surf/Tri.hh"

#include <cmath>

#include "corecel/Constants.hh"
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

    // Triangle vertex values
    Real3 a{0.0, 0.0, 0.0};
    Real3 b{1.0, 0, 0};
    Real3 c{0.5, 1.0, 1.0};
};

TEST_F(TriTest, construction)
{
    Tri<real_type> tri({a, b, c});

    EXPECT_VEC_SOFT_EQ(a, tri.verts()[0]);
    EXPECT_VEC_SOFT_EQ(b, tri.verts()[1]);
    EXPECT_VEC_SOFT_EQ(c, tri.verts()[2]);

    // Test that triangle is ordered CCW after construction
    Tri<real_type> tri_out_of_order({a, c, b});
    EXPECT_VEC_SOFT_EQ(a, tri.verts()[0]);
    EXPECT_VEC_SOFT_EQ(b, tri.verts()[1]);
    EXPECT_VEC_SOFT_EQ(c, tri.verts()[2]);
}

TEST_F(TriTest, calc_intersections)
{
    real_type cos45{std::cos(constants::pi / 4.0f)};

    Tri<real_type> tri({a, b, c});

    // Ray intersects triangle
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
    dir = {0.0f, -cos45, -cos45};
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
