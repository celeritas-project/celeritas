//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CubicSpline.test.cc
//---------------------------------------------------------------------------//
#include <cmath>
#include <vector>

#include "corecel/grid/Interpolator.hh"
#include "celeritas/grid/SplineDerivCalculator.hh"
#include "celeritas/grid/XsCalculator.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CubicSplineTest : public CalculatorTestBase
{
  protected:
    using Energy = XsCalculator::Energy;
    using VecReal = std::vector<real_type>;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CubicSplineTest, derivative_simple)
{
    VecReal x{0, 1, 2, 3, 4};
    VecReal y{0, 2, 1, 2, 0};

    auto result = SplineDerivCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(VecReal({-10.5, -3, 4.5, -3, -10.5}), result);
}

TEST_F(CubicSplineTest, derivative_constant)
{
    VecReal x{0, 1, 3, 7, 15};
    VecReal y{3, 3, 3, 3, 3};

    auto result = SplineDerivCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0, 0}), result);
}

TEST_F(CubicSplineTest, derivative_sin)
{
    size_type num_points = 10;
    VecReal x(num_points);
    VecReal y(num_points);

    for (size_type i = 0; i < num_points; ++i)
    {
        x[i] = i;
        y[i] = std::sin(i);
    }

    // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
    static double const expected_result[] = {
        -0.5225440890910731,
        -0.7736445427901106,
        -1.024744996489151,
        -0.1433986359548829,
        0.8198690326967976,
        1.038726849243206,
        0.3150069052469171,
        -0.757394547509858,
        -0.9096114092862184,
        -1.061828271062575,
    };
    auto result = SplineDerivCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(CubicSplineTest, derivative_nonuniform)
{
    VecReal x{0, 7, 16, 20, 24, 25, 29, 31, 38, 44, 53, 55, 60, 67, 74, 81};
    VecReal y{7, 3, 4, 2, 1, 1, 5, 3, 3, 2, 7, 1, 8, 7, 8, 2};

    // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
    static double const expected_result[] = {
        0.3495708556979281,
        0.108296016437501,
        -0.2019144911830484,
        0.1521114890387704,
        -0.03153146497203319,
        1.20686869356525,
        -1.509288867670116,
        0.6419958188901989,
        -0.3624781435261925,
        0.6550768332416019,
        -1.460455866973062,
        2.450502120449817,
        -0.9972235904702638,
        0.3462446731277701,
        -0.1428571428571429,
        -0.6319589588420556,
    };
    auto result = SplineDerivCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(CubicSplineTest, derivative_xs_grid)
{
    this->build({0.01, 100}, 5, [](real_type energy) {
        return energy >= 1 ? energy : 1 / energy;
    });
    this->convert_to_prime(3);

    static double const expected_result[] = {
        105520 / 33.0, 31880 / 11.0, -3160 / 33.0, -790 / 11.0, 5530 / 33.0};
    {
        auto result = SplineDerivCalculator(this->data(), this->values())();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
    {
        VecReal x{0.01, 0.1, 1, 10, 100};
        VecReal y{100, 10, 1, 10, 100};

        UniformGrid loge_grid(this->data().log_energy);
        XsCalculator calc_xs(this->data(), this->values());
        for (auto i : range(loge_grid.size()))
        {
            EXPECT_SOFT_EQ(x[i], std::exp(loge_grid[i]));
            EXPECT_SOFT_EQ(y[i], calc_xs[i]);
        }
        auto result = SplineDerivCalculator(make_span(x), make_span(y))();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST_F(CubicSplineTest, interpolate)
{
    VecReal x{0, 1, 2, 3, 4};
    VecReal y{0, 2, 1, 2, 0};
    auto y_prime = SplineDerivCalculator(make_span(x), make_span(y))();

    {
        SplineInterpolator interpolate({x[0], y[0], y_prime[0]},
                                       {x[1], y[1], y_prime[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.54875, interpolate(0.1));
        EXPECT_SOFT_EQ(1.84375, interpolate(0.5));
        EXPECT_SOFT_EQ(2.05875, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));
    }
    {
        SplineInterpolator interpolate({x[1], y[1], y_prime[1]},
                                       {x[2], y[2], y_prime[2]});
        EXPECT_EQ(2, interpolate(1));
        EXPECT_SOFT_EQ(1.40625, interpolate(1.5));
        EXPECT_SOFT_EQ(1.00000224875, interpolate(1.999));
        EXPECT_EQ(1, interpolate(2));
    }
}

TEST_F(CubicSplineTest, calculator)
{
    // x = [0.01, 0.1, 1, 10, 100], y = [100, 10, 1, 10, 100}]
    auto reference_xs
        = [](real_type energy) { return energy >= 1 ? energy : 1 / energy; };
    this->build({0.01, 100}, 5, reference_xs, true);
    this->convert_to_prime(3);

    XsCalculator calc_xs(this->data(), this->values());
    EXPECT_SOFT_EQ(10, calc_xs(Energy(0.1)));
    EXPECT_SOFT_EQ(-62.572615039281715, calc_xs(Energy(0.2)));
    EXPECT_SOFT_EQ(1, calc_xs(Energy(1)));
    EXPECT_SOFT_EQ(847.3120089786757, calc_xs(Energy(5)));
    EXPECT_SOFT_EQ(60.498378344017667, calc_xs(Energy(99.99)));
    EXPECT_SOFT_EQ(100, calc_xs(Energy(100)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
