//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/PolyEvaluator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/PolyEvaluator.hh"

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
    auto eval_poly = PolyEvaluator{1.0, 3, 4.0f, 5};
    EXPECT_TRUE(
        (std::is_same<decltype(eval_poly), PolyEvaluator<double, 3>>()));
    EXPECT_EQ(4 * sizeof(double), sizeof(eval_poly));
    EXPECT_DOUBLE_EQ(63.0, eval_poly(2.0));

    // Second-order int poly: 3 x^2 + 1
    auto eval_int_poly = PolyEvaluator{1, 0, 3};
    EXPECT_TRUE(
        (std::is_same<decltype(eval_int_poly), PolyEvaluator<int, 2>>()));
    EXPECT_EQ(3 * sizeof(int), sizeof(eval_int_poly));
    EXPECT_EQ(13, eval_int_poly(2));

    // First-order poly from an array
    constexpr Array<int, 2> linear_data{10, 1};
    constexpr auto eval_linear = PolyEvaluator(linear_data);
    EXPECT_TRUE(
        (std::is_same<decltype(eval_linear), PolyEvaluator<int, 1> const>()));
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
