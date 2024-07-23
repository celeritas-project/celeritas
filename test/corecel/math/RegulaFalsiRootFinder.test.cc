//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/RegulaRootFinder.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/RegulaFalsiRootFinder.hh"

#include <cmath>
#include <functional>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

#include "celeritas_test.hh"

namespace celeritas
{
using constants::pi;

namespace test
{

// Solve: (x-2)(x+2) = 0
TEST(RegulaFalsi, root_two)
{
    auto root = [](real_type t) { return (t - 2) * (t + 2); };

    constexpr real_type tol = SoftEqual<>{}.rel();

    RegulaFalsi find_root{root, tol};

    EXPECT_SOFT_EQ(2.0, find_root(1.0, 3.0));
    EXPECT_SOFT_EQ(-2.0, find_root(-3.0, -1.0));
}

// Solve: x^2 - x - 1 = 0
TEST(RegulaFalsi, golden_ratio)
{
    auto root = [](real_type t) { return ipow<2>(t) - t - 1; };

    constexpr real_type tol = SoftEqual<>{}.rel();

    RegulaFalsi find_root{root, tol};

    EXPECT_SOFT_EQ(1.618033988749, find_root(1.0, 2.0));
    EXPECT_SOFT_EQ(-0.6180339887498, find_root(-1.0, 0.0));
}

// Solve first 3 roots: cos(x) = 0
TEST(RegulaFalsi, trigometric)
{
    auto root = [](real_type t) { return std::cos(t); };

    constexpr real_type tol = SoftEqual<>{}.rel();

    RegulaFalsi find_root{root, tol};

    EXPECT_SOFT_EQ(pi * 0.5, find_root(0, pi));
    EXPECT_SOFT_EQ(pi * 1.5, find_root(pi, 2 * pi));
    EXPECT_SOFT_EQ(pi * 2.5, find_root(2 * pi, 3 * pi));
}

/*!
 * Solve Itersect
 *
 * x(t) = t
 * y(t) = exp(t)
 *
 * Point (0,2)
 * Direction (0,-1)
 */
TEST(RegulaFalsi, expomential_intersect)
{
    double x = 0.0;
    double y = 2.0;
    double u = 0.0;
    double v = -1.0;

    auto root
        = [&](real_type t) { return u * std::exp(t) - v * t + v * x - u * y; };

    constexpr real_type tol = SoftEqual<>{}.rel();

    RegulaFalsi find_root{root, tol};

    EXPECT_SOFT_EQ(0.0, find_root(-0.5, 0.5));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
