//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceGridHash.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/SurfaceGridHash.hh"

#include <algorithm>

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class SurfaceGridHashTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}

    using key_type = SurfaceGridHash::key_type;
    static constexpr auto redundant = SurfaceGridHash::redundant();
};

TEST_F(SurfaceGridHashTest, insertion)
{
    real_type const grid_size{0.1};
    real_type const tol{1e-4};
    SurfaceGridHash calc_hash{grid_size, tol};

    real_type const grid_offset{grid_size / 2};

    key_type const bin_center = [&] {
        // Insert at the middle of the bin, where there shouldn't be collisions
        auto keys = calc_hash(SurfaceType::px, 0.0);
        EXPECT_NE(keys[0], keys[1]);
        EXPECT_EQ(keys[1], redundant);
        return keys[0];
    }();
    key_type const bin_right = [&] {
        // Insert at the middle of the next bin
        auto keys = calc_hash(SurfaceType::px, grid_size);
        EXPECT_NE(keys[0], keys[1]);
        EXPECT_EQ(keys[1], redundant);
        return keys[0];
    }();
    key_type const bin_left = [&] {
        // Insert at the middle of the previous bin
        auto keys = calc_hash(SurfaceType::px, -grid_size);
        EXPECT_NE(keys[0], keys[1]);
        EXPECT_EQ(keys[1], redundant);
        return keys[0];
    }();

    EXPECT_NE(bin_left, bin_center);
    EXPECT_NE(bin_center, bin_right);
    EXPECT_NE(bin_right, bin_left);

    {
        // Insert near the end of the center
        auto keys = calc_hash(SurfaceType::px, grid_offset - 0.5 * tol);
        EXPECT_EQ(keys[0], bin_center);
        EXPECT_EQ(keys[1], bin_right);
    }
    {
        // Insert near the end of the left
        auto keys = calc_hash(SurfaceType::px, -grid_offset - 0.5 * tol);
        EXPECT_EQ(keys[0], bin_left);
        EXPECT_EQ(keys[1], bin_center);
    }
    {
        // Insert with a different surface type
        auto keys = calc_hash(SurfaceType::py, 0.0);
        EXPECT_NE(keys[0], bin_center);
        EXPECT_EQ(keys[1], redundant);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
