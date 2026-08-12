//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/QuarticSolver.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Alg1010Solver.hh"
#include "corecel/math/FerrariSolver.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/math/detail/Alg1010Impl.hh"

#include "celeritas_test.hh"

#ifndef CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS
// Double roots are notable, but (currently) counterproductive to test.
#    define CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS 0
#endif

namespace celeritas
{
namespace test
{
using Real5 = Array<real_type, 5>;
using Real4 = Array<real_type, 4>;
using Comp4 = Array<detail::Complex, 4>;
using Real2 = Array<real_type, 2>;
using TwinReal4 = Array<Real2, 4>;
using Roots = Array<real_type, 4>;
static constexpr real_type practical_tolerance
    = std::is_same_v<real_type, double> ? 1e-10 : 1e-6;

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

using QuarticSolvers = ::testing::Types<FerrariSolver, Alg1010Solver>;
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
    if constexpr (CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS)
    {
        EXPECT_VEC_SOFT_EQ(make_roots({2.0}),
                           sorted(solve(Real5{1, 0, 0, 0, -16})));
    }
    // x^4 - 2*x^3 - 2*x^2 + 8
    // One double root at 2, two imag rooots
    if constexpr (CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS)
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
    // [1.000000, 250.353530, -40111.798938, -6982487.888858,
    // -174874197.079351] One positive, three negative; taken from runtime
    // demonstration
    {
        EXPECT_VEC_NEAR(make_roots({187.76045963}),
                        sorted(solve(Real5{1.000000,
                                           250.353530,
                                           -40111.798938,
                                           -6982487.888858,
                                           -174874197.079351})),
                        practical_tolerance);
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
    if constexpr (CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS)
    {
        EXPECT_VEC_SOFT_EQ(make_roots({1.0, 2.0}),
                           sorted(solve(Real5{1, -6, 13, -12, 4})));
    }
    // [1.000000, 19.534611, -48660.891842, -476217.617178, 57126150.652217]
    // Two positive, two negative; taken from runtime demonstration
    {
        EXPECT_VEC_NEAR(make_roots({30.11797881450676, 213.24217809742103}),
                        sorted(solve(Real5{1.000000,
                                           19.534611,
                                           -48660.891842,
                                           -476217.617178,
                                           57126150.652217})),
                        practical_tolerance);
    }
    // [1.000000, 39.057127, -47753.699175, -940008.633992, 54384055.574769]
    // Also two positive two negative from runtime demonstration
    {
        EXPECT_VEC_NEAR(make_roots({25.6335352, 207.19825152}),
                        sorted(solve(Real5{1.000000,
                                           39.057127,
                                           -47753.699175,
                                           -940008.633992,
                                           54384055.574769})),
                        practical_tolerance);
    }
}

TYPED_TEST(QuarticSolverTest, three_roots)
{
    TypeParam solve{};
    // x^4 - 7*x^3 + 17*x^2 - 17*x + 6
    // Double root at 1, two roots at 2, 3
    if constexpr (CELERITAS_TEST_CRITICAL_QUARTIC_ROOTS)
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
/*
 * Test cases from Orellana & De Michele Algorithm 1010.
 * These cases are denoted by number matching appearance in the paper.
 */

/*
 * Generates a set of coefficients from a set of roots, as in ODM's demos 1-22
 */
Real5 make_coeffs(Comp4 const& roots)
{
    auto [x1c, x2c, x3c, x4c] = roots;
    return Real5{
        1.0,
        ((x1c + x2c + x3c + x4c) * -1.0).real,
        (x1c * x2c + (x1c + x2c) * (x3c + x4c) + x3c * x4c).real,
        (x1c * x2c * (x3c + x4c) * -1.0 - x3c * x4c * (x1c + x2c)).real,
        (x1c * x2c * x3c * x4c).real};
}

Real5 make_coeffs(Real4 const& roots)
{
    auto [x1, x2, x3, x4] = roots;
    return Real5{1.0,
                 (x1 + x2 + x3 + x4) * -1.0,
                 x1 * x2 + (x1 + x2) * (x3 + x4) + x3 * x4,
                 x1 * x2 * (x3 + x4) * -1.0 - x3 * x4 * (x1 + x2),
                 x1 * x2 * x3 * x4};
}

Real4 strip_imag(Comp4 const& comp_roots)
{
    auto [x1c, x2c, x3c, x4c] = comp_roots;
    return {x1c.real, x2c.real, x3c.real, x4c.real};
}

Real2 split(detail::Complex value)
{
    return {value.real, value.imag};
}

TwinReal4 split(Comp4 const& comp_roots)
{
    auto [x1c, x2c, x3c, x4c] = comp_roots;
    return {split(x1c), split(x2c), split(x3c), split(x4c)};
}

/*
 * Flip first two and last two roots in lieu of sorting complexes
 */
Comp4 flip2(Comp4 const& comp_roots)
{
    auto [x, y, z, w] = comp_roots;
    return {z, w, x, y};
}

/*
 * Alternatingly swap every other root
 */
Comp4 alternate(Comp4 const& comp_roots)
{
    auto [x, y, z, w] = comp_roots;
    return {y, x, w, z};
}

Array<real_type, 8> full_split(Comp4 const& comp_roots)
{
    auto [x, y, z, w] = comp_roots;
    return {x.real, x.imag, y.real, y.imag, z.real, z.imag, w.real, w.imag};
}
/*
 * Harness for ODM tests
 */
class ODMTest : public testing::Test
{
  protected:
    using ctype = detail::Complex;
    ODMTest() : solve_{} {}

    Alg1010Solver solve_;
    ctype i_{0, 1};
};

TEST_F(ODMTest, case_1)
{
    Real4 expected = sorted({1E9, 1E6, 1E3, 1});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

TEST_F(ODMTest, case_2)
{
    Real4 expected = sorted({2.003, 2.002, 2.001, 2});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

TEST_F(ODMTest, case_3)
{
    Real4 expected = sorted({1E53, 1E50, 1E49, 1E47});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

TEST_F(ODMTest, case_4)
{
    Real4 expected = sorted({1E14, 2, 1, -1});
    Real4 actual
        = sorted(strip_imag(solve_.unfiltered_roots(make_coeffs(expected))));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_5)
{
    Real4 expected = sorted({-2E7, 1E7, 1, -1});
    Real4 actual
        = sorted(strip_imag(solve_.unfiltered_roots(make_coeffs(expected))));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_6)
{
    Comp4 expected{1E7, -1E6, 1 + i_, 1 - i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_7)
{
    Comp4 expected{-7, -4, -1E6 + i_ * 1E5, -1E6 - i_ * 1E5};
    printf("Case 7!\n");
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_8)
{
    Comp4 expected{1E8, 11, 1E3 + i_, 1E3 - i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_9)
{
    Comp4 expected{1E7 + i_ * 1E6, 1E7 - i_ * 1E6, 1 + 2 * i_, 1 - 2 * i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_10)
{
    Comp4 expected{1E4 + 3 * i_, 1E4 - 3 * i_, -7 + 1E3 * i_, -7 - 1E3 * i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_11)
{
    Comp4 expected = {1.001 + 4.998 * i_,
                      1.001 - 4.998 * i_,
                      1.000 + 5.001 * i_,
                      1.000 - 5.001 * i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(alternate(actual)));
}
TEST_F(ODMTest, case_12)
{
    Comp4 expected{1E3 + 3 * i_, 1E3 - 3 * i_, 1E3 + i_, 1E3 - i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected),
                       full_split(flip2(alternate(actual))));
}
TEST_F(ODMTest, case_13)
{
    Comp4 expected{2 + 1E4 * i_, 2 - 1E4 * i_, 1 + 1E3 * i_, 1 - 1E3 * i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_14)
{
    Real4 expected = sorted({1000, 1000, 1000, 1000});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_15)
{
    Real4 expected = sorted({1000, 1000, 1000, 1E-15});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_16)
{
    Comp4 expected{
        1E16 + i_ * 1E7, 1E16 - i_ * 1E7, 1 + 0.1 * i_, 1 - 0.1 * i_};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_17)
{
    Real4 expected = sorted({10000, 10001, 10010, 10100});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_18)
{
    Comp4 expected{
        4E5 + i_ * 3E2, 4E5 - i_ * 3E2, 3E4 + i_ * 7E3, 3E4 - i_ * 7E3};
    Comp4 actual = solve_.unfiltered_roots(make_coeffs(expected));
    EXPECT_VEC_SOFT_EQ(full_split(expected), full_split(flip2(actual)));
}
TEST_F(ODMTest, case_19)
{
    Real4 expected = sorted({1E44, 1E30, 1E30, 1.0});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_20)
{
    Real4 expected = sorted({1E14, 1E7, 1E7, 1.0});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_21)
{
    Real4 expected = sorted({1E15, 1E7, 1E7, 1.0});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}
TEST_F(ODMTest, case_22)
{
    Real4 expected = sorted({1E154, 1E152, 10.0, 1.0});
    Real4 actual = sorted(solve_(make_coeffs(expected)));
    EXPECT_VEC_SOFT_EQ(expected, actual);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
