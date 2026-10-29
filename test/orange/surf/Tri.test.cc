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
    using tri_type = float;
    using Intersections = Tri<tri_type>::Intersections;
    using Vertex = Tri<tri_type>::Vertex;
};

TEST_F(TriTest, construction)
{
    Vertex a = {0.0, 0.0, 0.0};
    Vertex b = {1.0, 0, 0};
    Vertex c = {0.5, 1.0, 1.0};
    Tri<tri_type> tri(a, b, c);

    Vertex a_gold = {0, 0, 0};
    Vertex b_gold = {1, 0, 0};
    Vertex c_gold = {0.5, 1, 1};
    EXPECT_VEC_SOFT_EQ(a_gold, tri.a());
    EXPECT_VEC_SOFT_EQ(b_gold, tri.b());
    EXPECT_VEC_SOFT_EQ(c_gold, tri.c());
}

TEST_F(TriTest, calc_intersections)
{
    Vertex a = {0.0, 0.0, 0.0};
    Vertex b = {1.0, 0, 0};
    Vertex c = {0.5, 1.0, 1.0};
    Tri<tri_type> tri(a, b, c);

    Vertex pos = {0.4, 0.4, 0};
    Vertex dir = {0, 0, 1};

    auto t = tri.calc_intersections(pos, dir, SurfaceState::off);
    EXPECT_SOFT_EQ(0.40000000596046, t[0]);

    // Starting on the intersection point pointing away yields no intersection
    pos = {0.4, 0.4, tri_type(t[0])};
    t = tri.calc_intersections(pos, dir, SurfaceState::on);
    EXPECT_SOFT_EQ(no_intersection(), t[0]);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
