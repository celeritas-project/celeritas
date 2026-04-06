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

TEST(SurfacePhysicsUtilsTest, local_surf_id)
{
    using SD = LocalDirection;

    EXPECT_EQ(LocalSurfaceId{0},
              local_surf_id(LocalPositionId{0}, SD::forward));
    EXPECT_EQ(LocalSurfaceId{0},
              local_surf_id(LocalPositionId{1}, SD::reverse));
    EXPECT_EQ(LocalSurfaceId{1},
              local_surf_id(LocalPositionId{1}, SD::forward));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(local_surf_id(LocalPositionId{0}, SD::reverse),
                     DebugError);
        EXPECT_THROW(local_surf_id(LocalPositionId{}, SD::forward), DebugError);
        EXPECT_THROW(local_surf_id(LocalPositionId{}, SD::reverse), DebugError);
    }
}

TEST(SurfacePhysicsUtilsTest, next_local_pos_id)
{
    using SD = LocalDirection;

    EXPECT_EQ(LocalPositionId{},
              next_local_pos_id(LocalPositionId{0}, SD::reverse));
    EXPECT_EQ(LocalPositionId{1},
              next_local_pos_id(LocalPositionId{0}, SD::forward));
    EXPECT_EQ(LocalPositionId{0},
              next_local_pos_id(LocalPositionId{1}, SD::reverse));
    EXPECT_EQ(LocalPositionId{2},
              next_local_pos_id(LocalPositionId{1}, SD::forward));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(next_local_pos_id(LocalPositionId{}, SD::forward),
                     DebugError);
    }
}

TEST(SurfacePhysicsUtilsTest, calc_subsurface_direction)
{
    using SD = LocalDirection;

    EXPECT_EQ(calc_subsurface_direction({0, 0, -1}, {0, 0, 1}), SD::forward);
    EXPECT_EQ(calc_subsurface_direction({0, 0, 1}, {0, 0, 1}), SD::reverse);
    EXPECT_EQ(calc_subsurface_direction({1, 0, 0}, {0, 0, 1}), SD::reverse);
    EXPECT_EQ(calc_subsurface_direction({0, 0, -1}, {0, 0, -1}), SD::reverse);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
