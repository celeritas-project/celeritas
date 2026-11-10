//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BumpUlp.test.cc
//---------------------------------------------------------------------------//
#include "geocel/BumpUlp.hh"

#include <limits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(BumpUlpTest, components)
{
    constexpr real_type eps = std::numeric_limits<real_type>::epsilon();

    // Each moving component changes by n ULP of its own magnitude, regardless
    // of the magnitude of the direction
    Real3 pos{1, -2, 3};
    bump_ulp(10, {1e-6, -1, 0}, &pos);
    EXPECT_EQ(1 + 10 * eps, pos[0]);
    EXPECT_EQ(-2 - 10 * (2 * eps), pos[1]);
    EXPECT_EQ(3, pos[2]);

    // ULP is smaller below a power of two
    pos = {1, 2048, 0};
    bump_ulp(10, {-1, 1, 1}, &pos);
    EXPECT_EQ(1 - 10 * (eps / 2), pos[0]);
    EXPECT_EQ(2048 + 10 * (2048 * eps), pos[1]);
    EXPECT_EQ(10 * std::numeric_limits<real_type>::denorm_min(), pos[2]);

    // No bump
    pos = {1, 2, 3};
    bump_ulp(0, {1, 1, 1}, &pos);
    EXPECT_VEC_EQ((Real3{1, 2, 3}), pos);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
