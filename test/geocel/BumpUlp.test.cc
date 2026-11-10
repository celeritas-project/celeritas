//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/BumpUlp.test.cc
//---------------------------------------------------------------------------//
#include "geocel/BumpUlp.hh"

#include <cmath>
#include <limits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Size of one ULP from a value toward the sign of a direction
real_type ulp_toward(real_type p, real_type dir)
{
    constexpr real_type inf = std::numeric_limits<real_type>::infinity();
    return std::fabs(std::nextafter(p, dir > 0 ? inf : -inf) - p);
}

TEST(BumpUlpTest, components)
{
    // A power of two moves exactly n ULP, regardless of the magnitude of the
    // direction
    Real3 pos{1, -2, 3};
    bump_ulp(1024, {1e-6, -1, 0}, &pos);
    EXPECT_EQ(1024 * ulp_toward(1, 1), pos[0] - 1);
    EXPECT_EQ(1024 * ulp_toward(-2, -1), -2 - pos[1]);
    EXPECT_EQ(3, pos[2]);

    // Other values move between n and 2n ULP of their own magnitude; moving
    // down from a power of two crosses into smaller ULPs
    real_type const x = 1.5;
    pos = {x, 2047, 1};
    bump_ulp(1024, {1, -1, -1}, &pos);
    EXPECT_EQ(1536 * ulp_toward(x, 1), pos[0] - x);
    EXPECT_EQ(2047 * ulp_toward(2047, -1), 2047 - pos[1]);
    EXPECT_EQ(2048 * ulp_toward(1, -1), 1 - pos[2]);

    // Zero components are not bumped
    pos = {0, 0, 0};
    bump_ulp(1024, {1, -1, 1}, &pos);
    EXPECT_VEC_EQ((Real3{0, 0, 0}), pos);

    // No bump
    pos = {1, 2, 3};
    bump_ulp(0, {1, 1, 1}, &pos);
    EXPECT_VEC_EQ((Real3{1, 2, 3}), pos);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
