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
TEST(RegulaFalsi, root_two)
{
    DiagnosticRealFunc f{[](real_type t) { return (t - 2) * (t + 2); }};
    RegulaFalsi find_root{f, tol};

    EXPECT_SOFT_EQ(2.0, find_root(1.0, 3.0));
    EXPECT_EQ(21, f.exchange_count());
    EXPECT_SOFT_EQ(-2.0, find_root(-3.0, -1.0));
    EXPECT_EQ(21, f.exchange_count());
}

// Solve: x^2 - x - 1 = 0
TEST(RegulaFalsi, golden_ratio)
{
    DiagnosticRealFunc f{[](real_type t) { return ipow<2>(t) - t - 1; }};
    RegulaFalsi find_root{f, tol};

    EXPECT_SOFT_EQ(1.618033988749, find_root(1.0, 2.0));
    EXPECT_EQ(17, f.exchange_count());
    EXPECT_SOFT_EQ(-0.6180339887498, find_root(-1.0, 0.0));
    EXPECT_EQ(17, f.exchange_count());
}

// Solve first 3 roots: cos(x) = 0
TEST(RegulaFalsi, trigometric)
{
    DiagnosticRealFunc f{[](real_type t) { return std::cos(t); }};
    RegulaFalsi find_root{f, tol};

    EXPECT_SOFT_EQ(pi * 0.5, find_root(0, pi));
    EXPECT_EQ(3, f.exchange_count());
    EXPECT_SOFT_EQ(pi * 0.5, find_root(0.5, 3.0));
    EXPECT_EQ(7, f.exchange_count());
    EXPECT_SOFT_EQ(pi * 1.5, find_root(pi, 2 * pi));
    EXPECT_EQ(3, f.exchange_count());
    EXPECT_SOFT_EQ(pi * 2.5, find_root(2 * pi, 3 * pi));
    EXPECT_EQ(3, f.exchange_count());
}

/*!
 * Solve exponential intersect.
 *
 * x(t) = t
 * y(t) = exp(t)
 *
 * Point (0,2)
 * Direction (0,-1)
 */
TEST(RegulaFalsi, exponential_intersect)
{
    double x = 0.0;
    double y = 2.0;
    double u = 0.0;
    double v = -1.0;

    DiagnosticRealFunc f{
        [&](real_type t) { return u * std::exp(t) - v * t + v * x - u * y; }};
    RegulaFalsi find_root{f, tol};

    EXPECT_SOFT_EQ(0.0, find_root(-0.5, 0.5));
    EXPECT_EQ(3, f.exchange_count());
    EXPECT_SOFT_EQ(0.0, find_root(-0.5, 20));
    EXPECT_EQ(3, f.exchange_count());
    EXPECT_SOFT_EQ(0.0, find_root(-3, 1));
    EXPECT_EQ(3, f.exchange_count());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
