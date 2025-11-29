//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuarticSolver.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "orange/surf/detail/FerrariSolver.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*
 * Template test harness for arbitrary quartic solvers
 */
template<typename MyQuarticSolver>
class QuarticSolverTest : public ::celeritas::test::Test
{
  public:
    using Intersections = Array<real_type, 4>;
    using Coeffs4 = Array<real_type, 4>;
    using Coeffs5 = Array<real_type, 5>;

    // template<class MQS = MyQuarticSolver>
    // static inline MQS solve_quartic = ;

    QuarticSolverTest() {}

    template<class MQS = MyQuarticSolver>
    MQS get_solver()
    {
        MQS solve_quartic;  // Default tolerance
        return solve_quartic;
    }

    Intersections fill_inf(std::initializer_list<real_type> const& up_to_four)
    {
        Intersections all_roots;
        all_roots.fill(no_intersection());
        int i = 0;
        for (auto val : up_to_four)
        {
            all_roots[i] = val;
            i++;
        }
        return all_roots;
    }

    void expect_softeq_list(Intersections const& expected,
                            Intersections const& actual)
    {
        ASSERT_EQ(expected.size(), actual.size());
        for (auto i : range(expected.size()))
        {
            EXPECT_SOFT_EQ(expected[i], actual[i]);
        }
    }

    void expect_surface_roots_from_coeffs(
        std::initializer_list<real_type> const& expected, Coeffs4 const& abcd)
    {
        auto [a, b, c, d] = abcd;
        auto x = this->get_solver().solve_general({a, b, c, d, 0},
                                                  SurfaceState::on);
        this->expect_softeq_list(fill_inf(expected), x);
    }

    void expect_nonsurface_roots_from_coeffs(
        std::initializer_list<real_type> const& expected, Coeffs5 const& abcde)
    {
        auto x = this->get_solver()(abcde);
        this->expect_softeq_list(fill_inf(expected), x);
    }
};

//---------------------------------------------------------------------------//

using QuarticSolvers = ::testing::Types<FerrariSolver>;
TYPED_TEST_SUITE(QuarticSolverTest, QuarticSolvers, );

/*
 * Test cases with all non-zero roots, i.e., the ray does not start on or close
 * to the surface
 */

TYPED_TEST(QuarticSolverTest, no_roots)
{
    // x^4 + 2*x^3 - 2.999998*x^2 - 3.999998*x + 4.000005000001
    // Four complex roots 1+-0.001i, -2+-0.001i
    {
        this->expect_nonsurface_roots_from_coeffs(
            {}, {1, 2, -2.999998, -3.999998, 4.000005000001});
    }
    // x^4 + x^3 - 2.999999*x^2 - 0.999997*x + 2.200002
    // Two negative real roots 2, 1, and two imaginary roots 1+-0.001i
    {
        this->expect_nonsurface_roots_from_coeffs(
            {}, {1, 1, -2.999999, -0.999997, 2.200002});
    }
    // x^4 + 10*x^3 + 35*x^2 + 50*x + 24
    // Four negative roots -1, -2, -3, -4
    {
        this->expect_nonsurface_roots_from_coeffs({}, {1, 10, 35, 50, 24});
    }
}

TYPED_TEST(QuarticSolverTest, one_root)
{
    // x^4 - 16
    // One quadruple root at 2 (Critically degenerate torus)
    {
        this->expect_nonsurface_roots_from_coeffs({2.0}, {1, 0, 0, 0, -16});
    }
    // x^4 - 2*x^3 - 2*x^2 + 8
    // One double root at 2, two imag rooots
    {
        this->expect_nonsurface_roots_from_coeffs({2.0}, {1, -2, -2, 0, 8});
    }
    // x^4 - 3*x^3 + 1.000001*x^2 + 2.999999*x - 2.000002
    // One root at 2, one negative root at -1, two imag roots
    {
        this->expect_nonsurface_roots_from_coeffs(
            {2.0}, {1, -3, 1.000001, 2.999999, -2.000002});
    }
}

TYPED_TEST(QuarticSolverTest, two_roots)
{
    // x^4 + x^3 - 5*x^2 - 7*x + 10
    // Two roots at 2, 1, two imaginary roots
    {
        this->expect_nonsurface_roots_from_coeffs({1.0, 2.0},
                                                  {1, 1, -5, -7, 10});
    }
    // x^4 - 6*x^3 + 13*x^2 - 12*x + 4
    // Double root at 1, double root at 2
    {
        this->expect_nonsurface_roots_from_coeffs({1.0, 2.0},
                                                  {1, -6, 13, -12, 4});
    }
}

TYPED_TEST(QuarticSolverTest, three_roots)
{
    // x^4 - 7*x^3 + 17*x^2 - 17*x + 6
    // Double root at 1, two roots at 2, 3
    {
        this->expect_nonsurface_roots_from_coeffs({1.0, 2.0, 3.0},
                                                  {1, -7, 17, -17, 6});
    }
    // x^4 - 5*x^3 + 5*x^2 + 5*x - 6
    // Three roots at 1, 2, 3, negative root at -1
    {
        this->expect_nonsurface_roots_from_coeffs({1.0, 2.0, 3.0},
                                                  {1, -5, 5, 5, -6});
    }
}

TYPED_TEST(QuarticSolverTest, four_roots)
{
    // x^4 - 10*x^3 + 35*x^2 - 50*x + 24
    // Four roots at 1, 2, 3, 4
    this->expect_nonsurface_roots_from_coeffs({1.0, 2.0, 3.0, 4.0},
                                              {1, -10, 35, -50, 24});
}

//---------------------------------------------------------------------------//
/*
 * Test cases with a root at 0, i.e., a ray from a point exactly on the surface
 */
TYPED_TEST(QuarticSolverTest, surf_zero_roots)
{
    // x^4 + 6*x^3 + 11*x^2 + 6*x
    // Surface, three negative roots at -1, -2, -3
    this->expect_surface_roots_from_coeffs({}, {1, 6, 11, 6});
    // x^4
    // Quadruple root at 0, to catch degenerate cases
    this->expect_surface_roots_from_coeffs({}, {1, 0, 0, 0});
    // x^4 + 4*x^3
    // Triple root at 3, one at -4, to catch degenerate cases
    this->expect_surface_roots_from_coeffs({}, {1, 4, 0, 0});
}

TYPED_TEST(QuarticSolverTest, surf_one_root)
{
    // x^4 + 3*x^3 + 1*x^2 + -5*x
    // Surface, one root at 1, two imaginary roots
    {
        this->expect_surface_roots_from_coeffs({1.0}, {1, 3, 1, -5});
    }
    // x^4 - 2*x^3 + x^2
    // Double root on surface, double root at 1
    {
        this->expect_surface_roots_from_coeffs({1.0}, {1, -2, 1, 0});
    }
}

TYPED_TEST(QuarticSolverTest, surf_two_roots)
{
    // x^4 - 3*x^3 + 2*x^2
    // Double root on surface, two roots at 1, 2
    this->expect_surface_roots_from_coeffs({1.0, 2.0}, {1, -3, 2, 0});
}

TYPED_TEST(QuarticSolverTest, surf_two_roots_one_double)
{
    // x^4 - 5*x^3 + 8*x^2 - 4*x
    // Surface, one double root at 2, one root at 1
    this->expect_surface_roots_from_coeffs({1.0, 2.0}, {1, -5, 8, -4});
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
