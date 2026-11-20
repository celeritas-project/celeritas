//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BumpPosition.test.cc
//---------------------------------------------------------------------------//
#include "geocel/BumpPosition.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(BumpPositionTest, components)
{
    // Every moving component moves by the full distance, regardless of the
    // magnitude of the direction
    Real3 pos{1, -2, 3};
    bump_position(0.5, {1e-6, -1, 0}, &pos);
    EXPECT_VEC_EQ((Real3{1.5, -2.5, 3}), pos);

    // Components at zero move by the same distance
    pos = {0, 0, 0};
    bump_position(0.25, {-1e-12, 1, -0.0}, &pos);
    EXPECT_VEC_EQ((Real3{-0.25, 0.25, 0}), pos);

    // No bump
    pos = {1, 2, 3};
    bump_position(0, {1, 1, 1}, &pos);
    EXPECT_VEC_EQ((Real3{1, 2, 3}), pos);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
