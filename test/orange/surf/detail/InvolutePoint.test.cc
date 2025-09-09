//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvolutePoint.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/InvolutePoint.hh"

#include "corecel/Constants.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
constexpr real_type pi{constants::pi};

//---------------------------------------------------------------------------//
// r_b = 1, a = 0, t = 0
TEST(involute, point_one)
{
    real_type r_b = 1;
    real_type a = 0;
    detail::InvolutePoint calc_point{r_b, a};
    auto point = calc_point(clamp_to_nonneg(0));

    EXPECT_SOFT_EQ(1, point[0]);
    EXPECT_SOFT_EQ(0, point[1]);
}

// r_b = 1, a = pi*0.5 , t = 0
TEST(involute, point_two)
{
    real_type r_b = 1;
    real_type a = pi * 0.5;
    detail::InvolutePoint calc_point{r_b, a};
    auto point = calc_point(clamp_to_nonneg(0));

    EXPECT_SOFT_EQ(0, point[0]);
    EXPECT_SOFT_EQ(1, point[1]);
}

// r_b = 1, a = pi*0.5 , t = pi
TEST(involute, point_three)
{
    real_type r_b = 1;
    real_type a = pi * 0.5;
    detail::InvolutePoint calc_point{r_b, a};
    auto point = calc_point(real_type{pi});

    EXPECT_SOFT_EQ(-pi, point[0]);
    EXPECT_SOFT_EQ(-1, point[1]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
