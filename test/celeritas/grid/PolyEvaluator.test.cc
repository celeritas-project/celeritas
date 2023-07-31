//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/PolyEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/PolyEvaluator.hh"

#include <type_traits>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(PolyEvaluatorTest, make_eval)
{
    // Third-order double poly
    auto eval_poly = make_poly_evaluator(1.0, 3, 4.0f, 5);
    EXPECT_TRUE(
        (std::is_same<decltype(eval_poly), PolyEvaluator<double, 3>>()));
    EXPECT_EQ(4 * sizeof(double), sizeof(eval_poly));
    EXPECT_DOUBLE_EQ(63.0, eval_poly(2.0));

    // Second-order int poly: 3 x^2 + 1
    auto eval_int_poly = make_poly_evaluator(1, 0, 3);
    EXPECT_TRUE(
        (std::is_same<decltype(eval_int_poly), PolyEvaluator<int, 2>>()));
    EXPECT_EQ(3 * sizeof(int), sizeof(eval_int_poly));
    EXPECT_EQ(13, eval_int_poly(2));

    // First-order poly from an array
    constexpr Array<int, 2> linear_data{10, 1};
    constexpr PolyEvaluator<int, 1> eval_linear{linear_data};
    EXPECT_EQ(9, eval_linear(-1));
    EXPECT_EQ(10, eval_linear(0));
    EXPECT_EQ(12, eval_linear(2));
}

TEST(PolyEvaluatorTest, degenerate)
{
    PolyEvaluator<int, 0> eval(42);
    EXPECT_EQ(42, eval(12));
    EXPECT_EQ(42, eval(12345));
}

TEST(PolyEvaluatorTest, eval_real)
{
    PolyEvaluator<double, 2> eval(1.5, 3, -0.5);
    for (double x : {-1.0, 0.0, 2.5})
    {
        EXPECT_DOUBLE_EQ(1.5 + 3 * x + -0.5 * x * x, eval(x));
    }
}

TEST(PolyEvaluatorTest, eval_int)
{
    PolyEvaluator<int, 3> eval(3, 2, 1, -1);
    for (int x : {-1, 0, 2, 8})
    {
        EXPECT_EQ(3 + 2 * x + x * x - x * x * x, eval(x));
    }
}

TEST(PolyEvaluatorTest, deduction)
{
    Array<int, 5> coeffs = {1, -3, 2, 7, 5};
    auto eval = PolyEvaluator{coeffs};
    for (int x : {0, 1})
    {
        EXPECT_EQ(1 + x * 11, eval(x));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
