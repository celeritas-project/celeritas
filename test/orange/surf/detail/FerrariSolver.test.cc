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
        using Coeffs3 = Array<real_type, 3>;
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

        void expect_dominant_cubic_root(real_type const& expected, Coeffs3 const& bcd)
        {
            FerrariSolver solve_quartic(1,1,1,1);
            real_type bigroot = solve_quartic.dominant_root_normalized_cubic(bcd[0], bcd[1], bcd[2]);
            EXPECT_SOFT_EQ(expected, bigroot)
        }
}

//---------------------------------------------------------------------------//
/*
 * Test cases with all non-zero roots, i.e., the ray does not start on or close to the surface
 */
TEST_F(FerrariSolverTest, no_roots)
{
    // x**4 + 2*x**3 - 2.999998*x**2 - 3.999998*x + 4.000005000001
    // Four complex roots 1+-0.001i, -2+-0.001i
    {
        expect_nonsurface_roots(
            Intersections(no_intersection(), no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, 2, -2.999998, -3.999998, 4.000005000001)
        );
    }
    // x**4 + x**3 - 2.999999*x**2 - 0.999997*x + 2.000002
    // Two negative real roots 2, 1, and two imaginary roots 1+-0.001i
    {
        expect_nonsurface_roots(
            Intersections(no_intersection(), no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, 1, -2.999999, -0.999997, 2.000002)
        );
    }
    // x**4 + 10*x**3 + 35*x**2 + 50*x + 24 
    // Four negative roots -1, -2, -3, -4
    {
        expect_nonsurface_roots(
            Intersections(no_intersection(), no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, 10, 35, 50, 24)
        );
    }
}

TEST_F(FerrariSolverTest, one_root)
{
    // x**4 - 16
    // One quadruple root at 2 (Critically degenerate torus)
    {
        expect_nonsurface_roots(
            Intersections(2.0, no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, 0, 0, 0, -16)
        );
    }
    // x**4 - 2*x**3 - 2*x**2 + 8
    // One double root at 2, two imag rooots
    {
        expect_nonsurface_roots(
            Intersections(2.0, no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, -2, -2, 0, 8)
        );
    }
    // x**4 - 3*x**3 + 1.000001*x**2 + 2.999999*x - 2.000002
    // One root at 2, one negative root at -1, two imag roots
    {
        expect_nonsurface_roots(
            Intersections(2.0, no_intersection(), no_intersection(), no_intersection()),
            Coeffs5(1, -3, 1.000001, 2.999999, -2.000002)
        );
    }
}

TEST_F(FerrariSolverTest, two_roots)
{
    // x**4 - 5*x**3 + 9.000001*x**2 - 7.000003*x + 2.000002
    // Two roots at 2, 1, two imaginary roots
    {
        expect_nonsurface_roots(
            Intersections(1.0, 2.0, no_intersection(), no_intersection()),
            Coeffs5(1, -5, 9.000001, -7.000003, 2.000002)
        );
    }
    // x**4 - 6*x**3 + 13*x**2 - 12*x + 4
    // Double root at 1, double root at 2
    {
        expect_nonsurface_roots(
            Intersections(1.0, 2.0, no_intersection(), no_intersection()),
            Coeffs5(1, -6, 13, -12, 4)
        );
    }
}

TEST_F(FerrariSolverTest, three_roots)
{
    // x**4 - 7*x**3 + 17*x**2 - 17*x + 6
    // Double root at 1, two roots at 2, 3
    {
        expect_nonsurface_roots(
            Intersections(1.0, 2.0, 3.0, no_intersection()),
            Coeffs5(1, -7, 17, -17, 6)
        );
    }
    // x**4 - 5*x**3 + 5*x**2 + 5*x - 6
    // Three roots at 1, 2, 3, negative root at -1
    {
        expect_nonsurface_roots(
            Intersections(1.0, 2.0, 3.0, no_intersection()),
            Coeffs5(1, -5, 5, 5, 6)
        );
    }
}

TEST_F(FerrariSolverTest, four_roots)
{
    // x**4 - 10*x**3 + 35*x**2 - 50*x + 24
    // Four roots at 1, 2, 3, 4
    expect_nonsurface_roots(
        Intersections(1.0, 2.0, 3.0, 4.0),
        Coeffs5(1, -10, 35, -50, 24)
    );
}

//---------------------------------------------------------------------------//
/*
 * Test cases with a root at 0, i.e., a ray from a point exactly on the surface
 */
TEST_F(FerrariSolverTest, surf_zero_roots)
{
    // x**4 + 6*x**3 + 11*x**2 + 6*x
    // Surface, three negative roots at -1, -2, -3
    expect_surface_roots(
        Intersections(no_intersection(), no_intersection(), no_intersection(), no_intersection()),
        Coeffs4(1, 6, 11, 6)
    );
}

TEST_F(FerrariSolverTest, surf_one_root)
{
    // x**4 + 5*x**3 + 9*x**2 + 5*x
    // Surface, one root at 1, two imaginary roots
    {
        expect_surface_roots(
            Intersections(1.0, no_intersection(), no_intersection(), no_intersection()),
            Coeffs4(1, 5, 9. 5)
        );
    }
    // x**4 - 2*x**3 + x**2
    // Double root on surface, double root at 1
    {
        expect_surface_roots(
            Intersections(1.0, no_intersection(), no_intersection(), no_intersection()),
            Coeffs4(1, -2, 1, 0)
        );
    }
}

TEST_F(FerrariSolverTest, surf_two_roots)
{
    // x**4 - 3*x**3 + 2*x**2
    // Double root on surface, two roots at 1, 2
    expect_surface_roots(
        Intersections(1.0, 2.0, no_intersection(), no_intersection()),
        Coeffs4(1, -3, 2, 0)
    );
}

TEST_F(FerrariSolverTest, surf_two_roots_one_double)
{
    // x**4 - 5*x**3 + 8*x**2 - 4*x
    // Surface, one double root at 2, one root at 1
    expect_surface_roots(
        Intersections(1.0, 2.0, no_intersection(), no_intersection()),
        Coeffs4(1, -5, 8, -4)
    );
}

//---------------------------------------------------------------------------//
/*
 * Test cases for the cubic function solver, which should always return the largest root 
 */
TEST_F(FerrariSolverTest, cubic_three_real){
    // x**3 - 6*x**2 + 11*x - 6
    // At 3, 2, 1, should find 3
    {
        expect_dominant_cubic_root(
            3.0,
            Coeffs3(-6, 11, -6)
        );
    }
    // x**3 + 4*x**2 + 3*x
    // At 0, -1, -3 should find 0
    {
        expect_dominant_cubic_root(
            0.0,
            Coeffs3(4, 3, 0)
        );
    }
    // x**3 + 6*x**2 + 11*x + 6
    // At -1, -2, -3 should find -1
    {
        expect_dominant_cubic_root(
            -1.0,
            Coeffs3(6, 11, 6)
        );
    }
    // x**3 + 301*x**2 + 298*x - 600
    // At 1, -2, -300 should find 1
    {
        expect_dominant_cubic_root(
            1.0,
            Coeffs3(301, 298, -600)
        );
    }
}

TEST_F(FerrariSolverTest, cubic_one_real) {
    // x**3 + 3*x**2 + x - 5
    // At 1 with two imag should find 1
    {
        expect_dominant_cubic_root(
            1.0, 
            Coeffs3(3, 1, -5)
        );
    }
    // x**3 - 3*x**2 + x + 5
    // At -1 with two imag should find -1
    {
        expect_dominant_cubic_root(
            -1,
            Coeffs3(-3, 1, 5)
        );
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
