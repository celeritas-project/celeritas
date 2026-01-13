//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/SurfacePhysicsUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(SurfacePhysicsUtilsTest, is_entering_surface)
{
    EXPECT_TRUE(is_entering_surface({0, 0, -1}, {0, 0, 1}));
    EXPECT_FALSE(is_entering_surface({0, 0, 1}, {0, 0, 1}));
    EXPECT_FALSE(is_entering_surface({1, 0, 0}, {0, 0, 1}));
    EXPECT_FALSE(is_entering_surface({0, 0, -1}, {0, 0, -1}));
}

TEST(SurfacePhysicsUtilsTest, next_subsurface_position)
{
    using SD = SubsurfaceDirection;

    EXPECT_EQ(SurfaceTrackPosition{2},
              next_subsurface_position(SurfaceTrackPosition{1}, SD::forward));
    EXPECT_EQ(SurfaceTrackPosition{0},
              next_subsurface_position(SurfaceTrackPosition{1}, SD::reverse));
    EXPECT_EQ(SurfaceTrackPosition{},
              next_subsurface_position(SurfaceTrackPosition{0}, SD::reverse));
}

TEST(SurfacePhysicsUtilsTest, calc_subsurface_direction)
{
    using SD = SubsurfaceDirection;

    EXPECT_EQ(calc_subsurface_direction({0, 0, -1}, {0, 0, 1}), SD::forward);
    EXPECT_EQ(calc_subsurface_direction({0, 0, 1}, {0, 0, 1}), SD::reverse);
    EXPECT_EQ(calc_subsurface_direction({1, 0, 0}, {0, 0, 1}), SD::reverse);
    EXPECT_EQ(calc_subsurface_direction({0, 0, -1}, {0, 0, -1}), SD::reverse);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
