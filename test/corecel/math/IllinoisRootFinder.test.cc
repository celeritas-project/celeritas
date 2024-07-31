//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/IllinoisRootFinder.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/IllinoisRootFinder.hh"

#include <cmath>
#include <functional>

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/math/SoftEqual.hh"
#include "orange/OrangeTypes.hh"

#include "DiagnosticRealFunc.hh"
#include "celeritas_test.hh"

using celeritas::constants::pi;
inline constexpr auto tol = celeritas::SoftEqual<>{}.rel();

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

// Solve: (x-2)(x+2) = 0
TEST(Illinois, root_two)
{
    DiagnosticRealFunc f{[](real_type t) { return (t - 2) * (t + 2); }};

    IllinoisRootFinder find_root{f, tol};

    EXPECT_SOFT_EQ(2.0, find_root(1.75, 2.25));
    EXPECT_SOFT_EQ(-2.0, find_root(-2.25, -1.75));
}

// Solve: x^2 - x - 1 = 0
TEST(Illinois, golden_ratio)
{
    DiagnosticRealFunc f{[](real_type t) { return ipow<2>(t) - t - 1; }};

    IllinoisRootFinder find_root{f, tol};

    EXPECT_SOFT_EQ(1.618033988749, find_root(1.5, 1.75));
    EXPECT_SOFT_EQ(-0.6180339887498, find_root(-0.75, -0.5));
    EXPECT_EQ(if_double_else(18, 12), f.exchange_count());
}

// Solve first 3 roots: cos(x) = 0
TEST(Illinois, trigometric)
{
    DiagnosticRealFunc f{[](real_type t) { return std::cos(t); }};

    IllinoisRootFinder find_root{f, tol};

    EXPECT_SOFT_EQ(pi * 0.5, find_root(0, pi));
    EXPECT_SOFT_EQ(pi * 1.5, find_root(pi, 2 * pi));
    EXPECT_SOFT_EQ(pi * 2.5, find_root(2 * pi, 3 * pi));
}

/*!
 * Solve exponential intersect.
 *
 * x(t) = t
 * y(t) = exp(t-1)
 *
 * Point (1.5,0.5)
 * Direction (-0.7071067812,0.7071067812)
 */
TEST(Illinois, exponential_intersect)
{
    real_type x = 1.5;
    real_type y = 0.5;
    real_type u = -0.7071067812;
    real_type v = 0.7071067812;

    DiagnosticRealFunc f{[&](real_type t) {
        return u * std::exp(t - 1) - v * t + v * x - u * y;
    }};

    IllinoisRootFinder find_root{f, tol};

    EXPECT_SOFT_EQ(1.0, find_root(-0.5, 0.5));
    EXPECT_EQ(if_double_else(10, 7), f.exchange_count());
    EXPECT_SOFT_EQ(1.0, find_root(0.5, 1.5));
    EXPECT_EQ(if_double_else(9, 7), f.exchange_count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
