//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/QuarticSolver.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/FerrariSolver.hh"
#include "corecel/math/NumericLimits.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using Real5 = Array<real_type, 5>;
using Real4 = Array<real_type, 4>;
using Roots = Array<real_type, 4>;

//---------------------------------------------------------------------------//
/*!
 * Fills a list of fewer than 4 roots with "no real positive root"
 */
Roots make_roots(std::initializer_list<real_type> const& inp)
{
    CELER_EXPECT(inp.size() <= Roots{}.size());
    Roots result;
    auto iter = std::copy(inp.begin(), inp.end(), result.begin());
    std::fill(iter, result.end(), NumericLimits<real_type>::infinity());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sorts a given array of four roots and returns the array.
 */
Roots sorted(Roots four_roots)
{
    sort(four_roots.begin(), four_roots.end());
    return four_roots;
}

//---------------------------------------------------------------------------//
/*
 * Template test harness for arbitrary quartic solvers
 */
template<typename MyQuarticSolver>
class QuarticSolverTest : public ::celeritas::test::Test
{
};

using QuarticSolvers = ::testing::Types<FerrariSolver>;
TYPED_TEST_SUITE(QuarticSolverTest, QuarticSolvers, );

//---------------------------------------------------------------------------//
/*
 * Test cases with all non-zero roots, i.e., the ray does not start on or close
 * to the surface
 */

TYPED_TEST(QuarticSolverTest, no_roots)
{
    TypeParam solve{};
    // x^4 + 2*x^3 - 2.98*x^2 - 3.98*x + 4.0501
    // Four complex roots 1+-0.1i, -2+-0.1i
    {
        EXPECT_VEC_SOFT_EQ(make_roots({}),
                           sorted(solve(Real5{1, 2, -2.98, -3.98, 4.0501})));
    }
    // x^4 + x^3 - 2*x^2 + 2*x + 4
    // Two negative real roots 2, 1, and two imaginary roots 1+-i
    {
        EXPECT_VEC_SOFT_EQ(make_roots({}),
                           sorted(solve(Real5{1, 1, -2, 2, 4})));
    }
    // x^4 + 10*x^3 + 35*x^2 + 50*x + 24
    // Four negative roots -1, -2, -3, -4
    {
        EXPECT_VEC_SOFT_EQ(make_roots({}),
                           sorted(solve(Real5{1, 10, 35, 50, 24})));
    }
}

TYPED_TEST(QuarticSolverTest, one_root)
{
    TypeParam solve{};
    // x^4 - 16
    // One quadruple root at 2 (Critically degenerate torus)
    {
        EXPECT_VEC_SOFT_EQ(make_roots({2.0}),
                           sorted(solve(Real5{1, 0, 0, 0, -16})));
    }
    // x^4 - 2*x^3 - 2*x^2 + 8
    // One double root at 2, two imag rooots
    {
        EXPECT_VEC_SOFT_EQ(make_roots({2.0}),
                           sorted(solve(Real5{1, -2, -2, 0, 8})));
    }
    // x^4 - 3*x^3 + 2*x^2 + 2x - 4
    // One root at 2, one negative root at -1, two imag roots
    {
        EXPECT_VEC_SOFT_EQ(make_roots({2.0}),
                           sorted(solve(Real5{1, -3, 2, 2, -4})));
    }
}

TYPED_TEST(QuarticSolverTest, two_roots)
{
    TypeParam solve{};
    // x^4 + x^3 - 5*x^2 - 7*x + 10
    // Two roots at 2, 1, two imaginary roots
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0}),
                           sorted(solve(Real5{1, 1, -5, -7, 10})));
    }
    // x^4 - 6*x^3 + 13*x^2 - 12*x + 4
    // Double root at 1, double root at 2
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0}),
                           sorted(solve(Real5{1, -6, 13, -12, 4})));
    }
}

TYPED_TEST(QuarticSolverTest, three_roots)
{
    TypeParam solve{};
    // x^4 - 7*x^3 + 17*x^2 - 17*x + 6
    // Double root at 1, two roots at 2, 3
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0, 3.0}),
                           sorted(solve(Real5{1, -7, 17, -17, 6})));
    }
    // x^4 - 5*x^3 + 5*x^2 + 5*x - 6
    // Three roots at 1, 2, 3, negative root at -1
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0, 3.0}),
                           sorted(solve(Real5{1, -5, 5, 5, -6})));
    }
}

TYPED_TEST(QuarticSolverTest, four_roots)
{
    TypeParam solve{};
    // x^4 - 10*x^3 + 35*x^2 - 50*x + 24
    // Four roots at 1, 2, 3, 4
    EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0, 3.0, 4.0}),
                       sorted(solve(Real5{1, -10, 35, -50, 24})));
}

//---------------------------------------------------------------------------//
/*
 * Test cases with a root at 0, i.e., a ray from a point exactly on the surface
 */
TYPED_TEST(QuarticSolverTest, surf_zero_roots)
{
    TypeParam solve{};
    // x^4 + 6*x^3 + 11*x^2 + 6*x
    // Surface, three negative roots at -1, -2, -3
    EXPECT_VEC_SOFT_EQ(make_roots({}), sorted(solve(Real4{1, 6, 11, 6})));
}

TYPED_TEST(QuarticSolverTest, surf_one_root)
{
    TypeParam solve{};
    // x^4 + 3*x^3 + 1*x^2 + -5*x
    // Surface, one root at 1, two imaginary roots
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0}),
                           sorted(solve(Real4{1, 3, 1, -5})));
    }
    // x^4 + 3*x^3 - 4*x
    // Surface, one root at 1, two roots at -2
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0}),
                           sorted(solve(Real4{1, 3, 0, -4})));
    }
}

TYPED_TEST(QuarticSolverTest, surf_two_roots)
{
    TypeParam solve{};
    // x^4 - 2*x^3 - x^2 + 2*x
    // Surface, two roots at 1, 2, one root at -1
    EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0}),
                       sorted(solve(Real4{1, -2, -1, 2})));
}

TYPED_TEST(QuarticSolverTest, surf_three_roots)
{
    TypeParam solve{};
    // x^4 - 6*x^3 + 11*x^2 - 6*x
    // Surface, roots at 1, 2, and 3
    EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0, 3.0}),
                       sorted(solve(Real4{1, -6, 11, -6})));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
