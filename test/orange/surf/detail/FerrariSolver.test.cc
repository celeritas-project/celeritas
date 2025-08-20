
//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/FerrariSolver.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/FerrariSolver.hh"

#include "celeritas_test.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*
 * Test Harness for FerrariSolver
 */
class FerrariSolverTest : public ::celeritas::test::Test
{
    public:
        using FS = FerrariSolver;
        using Intersections = Array<real_type, 4>;
        using Coeffs4 = Array<real_type, 4>;
        using Coeffs5 = Array<real_type, 5>;

        FerrariSolverTest(){}

        void expect_softeq_list(Intersections const& expected, Intersections const& actual)
        {
            ASSERT_EQ(expected.size(), actual.size());
            for (auto i : range(expected.size()))
            {
                EXPECT_SOFT_EQ(expected[i], actual[i])
            }
        }

        void expect_surface_roots(Intersections const& expected, Coeffs4 const& abcd)
        {
            FerrariSolver solve_quartic(abcd[0], abcd[1], abcd[2], abcd[3]);
            auto x = solve_quartic();
            expect_softeq_list(expected, x);
        }

        void expect_nonsurface_roots(Intersections const& expected, Coeffs5 const& abcde)
        {
            FerrariSolver solve_quartic(abcde[0], abcde[1], abcde[2], abcde[3]);
            auto x = solve_quartic(abcde[4]);
            expect_softeq_list(expected, x);
        }
}

TEST(SolveNonsurface, no_roots)
{
    // x**4 + 2*x**3 - 2.999998*x**2 - 3.999998*x + 4.000005000001
    // Four complex roots 1+-0.001i, -2+-0.001i
    {
        double b = 2;
        double c = -2.999998;
        double d = -3.999998;
        double e = 4.000005000001;

        FerrariSolver solve_quartic(1, b, c, d);
        auto x = solve_quartic(e);

        EXPECT_SOFT_EQ(no_intersection(), x[0]);
        EXPECT_SOFT_EQ(no_intersection(), x[1]);
        EXPECT_SOFT_EQ(no_intersection(), x[2]);
        EXPECT_SOFT_EQ(no_intersection(), x[3]);
    }
    // Two negative real roots 2, 1, and two imaginary roots 1+-0.001i

    // Four negative roots -1, -2, -3, -4
}

TEST(SolveNonsurface, one_root) 
{
    // x**4 - 16
    // One quadruple root at 2 (Critically degenerate torus)

    // x**4 - 2*x**3 - 2*x**2 + 8
    // One double root at 2, two imag rooots
    {
        double b = -2;
        double c = -2;
        double d = 0;
        double e = 8;

        FerrariSolver solve_quartic(1, b, c, d);
        auto x = solve_quartic(e);

        EXPECT_SOFT_EQ(2.0, x[0]);
        EXPECT_SOFT_EQ(no_intersection(), x[1]);
        EXPECT_SOFT_EQ(no_intersection(), x[2]);
        EXPECT_SOFT_EQ(no_intersection(), x[3]);
    }

    // One root at 2, one negative root at -1, two imag roots
}

TEST(SolveNonsurface, two_roots)
{
    // Two roots at 2, 1, two negative roots at -3, -4

    // Two roots at 2, 1, two imaginary roots

    // Double root at 1, double root at 2
}

TEST(SolveNonsurface, three_roots)
{
    // Double root at 1, two roots at 2, 3

    // Three roots at 1, 2, 3, negative root at -1
}

TEST(SolveNonsurface, four_roots)
{
    // Four roots at 1, 2, 3, 4
}

TEST(SolveSurface, zero_roots)
{
    // Surface, three negative roots at -1, -2, -3
}

TEST(SolveSurface, one_root)
{
    // Surface, one root at 1, two imaginary roots

    // Double root on surface, double root at 1
}

TEST(SolveSurface, two_roots)
{
    // Double root on surface, two roots at 1, 2
}

TEST(SolveSurface, two_roots_one_double)
{
    // Surface, one double root at 2, one root at 1
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
