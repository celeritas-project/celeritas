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
    using tri_type = double;
    using Intersections = Tri<tri_type>::Intersections;
    using Real3 = Tri<tri_type>::Real3;
};

TEST_F(TriTest, construction)
{
    Real3 a = {0.0, 0.0, 0.0};
    Real3 b = {1.0, 0, 0};
    Real3 c = {0.5, 1.0, 1.0};
    Tri<tri_type> tri({a, b, c});

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
    Tri<tri_type> tri({a, b, c});

    Real3 pos = {0.4, 0.4, 0};
    Real3 dir = {0, 0, 1};

    auto t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(0.40000000, t[0]);

    // Starting on the intersection point pointing away yields no intersection
    pos = {0.4, 0.4, tri_type(t[0])};
    t = tri.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), t[0]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
