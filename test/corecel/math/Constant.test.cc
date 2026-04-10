//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/Constant.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/Constant.hh"

#include <type_traits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(ConstantTest, comparison)
{
    constexpr Constant pi{3.14};
    EXPECT_TRUE(pi > 3);
    EXPECT_TRUE(pi > 3.0);
    EXPECT_TRUE(pi > 3.0f);
    EXPECT_TRUE(pi > 3_C);
    EXPECT_FALSE(pi == 3);
    EXPECT_FALSE(pi == 3.0);
    EXPECT_FALSE(pi == 3.0f);
    EXPECT_FALSE(pi == 3_C);
    EXPECT_TRUE(pi != 3);
    EXPECT_TRUE(pi != 3_C);
    EXPECT_TRUE(pi == 3.14);
    EXPECT_TRUE(pi == 3.14f);
    EXPECT_TRUE(pi == 3.14_C);
    EXPECT_TRUE(pi < 4);
    EXPECT_TRUE(pi < 4.0);
    EXPECT_TRUE(pi < 4.0f);
    EXPECT_TRUE(pi < 4_C);
}

TEST(ConstantTest, multiplication)
{
    constexpr Constant pi{3.14};
    {
        auto twopi = 2 * pi;
        EXPECT_TRUE((std::is_same_v<Constant, decltype(twopi)>));
    }

    {
        auto halfpi = 0.5 * pi;
        EXPECT_TRUE((std::is_same_v<double, decltype(halfpi)>));
    }

    {
        auto halfpi = 0.5f * pi;
        EXPECT_TRUE((std::is_same_v<float, decltype(halfpi)>));
    }

    {
        auto halfpi = 0.5_C * pi;
        EXPECT_TRUE((std::is_same_v<Constant, decltype(halfpi)>));
    }

    {
        // Multiplication is performed at single precision,
        // then promoted to double
        auto expr = 1.23f * pi * 2 + 4.0;
        EXPECT_TRUE((std::is_same_v<double, decltype(expr)>));
        EXPECT_EQ(1.23f * 3.14f * 2 + 4.0, expr);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
