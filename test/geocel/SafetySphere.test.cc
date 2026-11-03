//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SafetySphere.test.cc
//---------------------------------------------------------------------------//
#include "geocel/SafetySphere.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(SafetySphereTest, empty)
{
    SafetySphere sphere;
    EXPECT_FALSE(sphere);
    EXPECT_FALSE(sphere.contains({0, 0, 0}, 0));
    // The default center isn't a calculated safety
    EXPECT_FALSE(sphere.is_centered_on({0, 0, 0}));
}

TEST(SafetySphereTest, zero_radius)
{
    // Zero safety is a calculated value, but the sphere still contains nothing
    SafetySphere sphere{{1, 2, 3}, 0};
    EXPECT_FALSE(sphere);
    EXPECT_FALSE(sphere.contains({1, 2, 3}, 0));
    EXPECT_TRUE(sphere.is_centered_on({1, 2, 3}));
    EXPECT_FALSE(sphere.is_centered_on({1, 2, 4}));
}

TEST(SafetySphereTest, contains)
{
    SafetySphere sphere{{1, 2, 3}, 2};
    EXPECT_TRUE(sphere);

    // Center, with tolerances below and at the radius
    EXPECT_TRUE(sphere.contains({1, 2, 3}, 0));
    EXPECT_TRUE(sphere.contains({1, 2, 3}, 1.5));
    EXPECT_FALSE(sphere.contains({1, 2, 3}, 2));

    // Off center along x: 1.5 from the center
    EXPECT_TRUE(sphere.contains({2.5, 2, 3}, 0.5));
    EXPECT_FALSE(sphere.contains({2.5, 2, 3}, 0.6));

    // On and just outside the surface
    EXPECT_TRUE(sphere.contains({1, 2, 1}, 0));
    EXPECT_FALSE(sphere.contains({1, 2, 0.9}, 0));

    // Diagonal: distance sqrt(3) < 2
    EXPECT_TRUE(sphere.contains({2, 3, 4}, 0.2));
    EXPECT_FALSE(sphere.contains({2, 3, 4}, 0.3));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
