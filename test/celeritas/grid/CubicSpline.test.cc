//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/CubicSpline.test.cc
//---------------------------------------------------------------------------//
#include <cmath>
#include <vector>

#include "corecel/grid/Interpolator.hh"
#include "corecel/grid/VectorUtils.hh"
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
    using BC = SplineDerivCalculator::BoundaryCondition;
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
    {
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(VecReal({-10.5, -3, 4.5, -3, -10.5}), result);
    }
    {
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, -6, 6, -6, 0}), result);
    }
}

TEST_F(CubicSplineTest, derivative_constant)
{
    VecReal x{0, 1, 3, 7, 15};
    VecReal y{3, 3, 3, 3, 3};
    {
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0, 0}), result);
    }
    {
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0, 0}), result);
    }
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
    auto result
        = SplineDerivCalculator(make_span(x), make_span(y), BC::not_a_knot)();
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(CubicSplineTest, derivative_nonuniform)
{
    VecReal x{0, 7, 16, 20, 24, 25, 29, 31};
    VecReal y{13, 12, 10, 2, 5, 8, 12, 15};
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
        static double const expected_result[] = {
            4.9426958709655300e-01,
            3.3921993080859636e-02,
            -5.5795348493931729e-01,
            8.8370650100696158e-01,
            1.1481274809114712e+00,
            -1.5161008131425509e+00,
            5.0322016262851021e-01,
            1.5128806505140406e+00,
        };
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='natural'
        static double const expected_result[] = {
            2.7755575615628914e-17,
            1.5412058764458358e-01,
            -6.0089436453523881e-01,
            8.9237538061207400e-01,
            1.1563928420869445e+00,
            -1.6334299433177244e+00,
            7.9447664777257454e-01,
            -1.1102230246251565e-16,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST_F(CubicSplineTest, derivative_log)
{
    // Trimmed energy loss grid
    VecReal x = logspace(1e-4, 1e7, 12);
    VecReal y = {
        839.668353354807,
        430.530096695467,
        111.600220710967,
        22.6117194229536,
        10.6619173294951,
        11.0069268409596,
        11.3553238163283,
        11.3784262549454,
        11.378228777509,
        11.3782267757997,
        11.3782267557938,
        11.3782267555937,
    };
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
        static double const expected_result[] = {
            1.3572120032350275e+08,
            1.2296066842786135e+08,
            -4.6446505285416190e+06,
            2.1869043745504515e+05,
            -1.0150685763332291e+04,
            4.7134219534104602e+02,
            -2.1886550971249481e+01,
            1.0161669665900670e+00,
            -4.6912246500744756e-02,
            1.5902456440930928e-03,
            1.1926842330698196e-03,
            -2.7829298771629126e-03,
        };
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='natural'
        static double const expected_result[] = {
            0,  // Note: scipy returns a non-zero value for the first f''
            1.2926283143071672e+08,
            -4.9372891024729721e+06,
            2.3227899981848482e+05,
            -1.0781665569764382e+04,
            5.0064153314767589e+02,
            -2.3247113502625947e+01,
            1.0794707549553146e+00,
            -5.0124327766643099e-02,
            2.3264455925446851e-03,
            -1.0574752693384931e-04,
            0,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
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
        auto result = SplineDerivCalculator(
            this->data(), this->values(), BC::not_a_knot)();
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
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST_F(CubicSplineTest, interpolate)
{
    VecReal x{0, 1, 2, 3, 4};
    VecReal y{0, 2, 1, 2, 0};

    auto y_prime
        = SplineDerivCalculator(make_span(x), make_span(y), BC::not_a_knot)();
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
    y_prime = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
    {
        SplineInterpolator interpolate({x[0], y[0], y_prime[0]},
                                       {x[1], y[1], y_prime[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.299, interpolate(0.1));
        EXPECT_SOFT_EQ(1.375, interpolate(0.5));
        EXPECT_SOFT_EQ(1.971, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));
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
