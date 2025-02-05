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
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, -6, 6, -6, 0}), result);
    }
    {
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(VecReal({-10.5, -3, 4.5, -3, -10.5}), result);
    }
    {
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::geant)();
        EXPECT_VEC_SOFT_EQ(VecReal({-4.3125, 0, 4.3125, -3, -10.3125}), result);
    }
}

TEST_F(CubicSplineTest, derivative_constant)
{
    VecReal x{0, 1, 3, 7, 15};
    VecReal y{3, 3, 3, 3, 3};
    {
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0, 0}), result);
    }
    {
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0, 0}), result);
    }
    {
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::geant)();
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
        x[i] = static_cast<real_type>(i);
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
        // Values from scipy.interpolate.CubicSpline with bc_type='natural'
        static double const expected_result[] = {
            2.775557561562891e-17,
            0.1541205876445836,
            -0.6008943645352388,
            0.892375380612074,
            1.156392842086944,
            -1.633429943317724,
            0.7944766477725745,
            -1.110223024625157e-16,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
        static double const expected_result[] = {
            0.494269587096553,
            0.03392199308085964,
            -0.5579534849393173,
            0.8837065010069616,
            1.148127480911471,
            -1.516100813142551,
            0.5032201626285102,
            1.512880650514041,
        };
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
    {
        static double const expected_result[] = {
            0.51145818598167,
            0.042461742420306,
            -0.56053368501573,
            0.89527633099057,
            1.1096028682332,
            -1.5184630831654,
            0.54085090248942,
            1.5705078953168,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::geant)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

TEST_F(CubicSplineTest, derivative_log)
{
    // Trimmed energy loss grid
    VecReal x
        = {1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7};
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
        // Values from scipy.interpolate.CubicSpline with bc_type='natural'
        // Note: scipy returns a non-zero value for the first y'' (O(1e-6))
        static double const expected_result[] = {
            0,
            129262831.4307167,
            -4937289.102472972,
            232278.9998184848,
            -10781.66556976438,
            500.6415331476759,
            -23.24711350262595,
            1.079470754955315,
            -0.0501243277666431,
            0.002326445592544685,
            -0.0001057475269338493,
            6.776263578034403e-21,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);

        SplineInterpolator interpolate({x[0], y[0], result[0]},
                                       {x[1], y[1], result[1]});
        EXPECT_SOFT_EQ(834.92850241567407, interpolate(1.1e-4));
        EXPECT_SOFT_EQ(651.6053622151029, interpolate(5e-4));
    }
    {
        // Values from scipy.interpolate.CubicSpline with bc_type='not-a-knot'
        static double const expected_result[] = {
            135721200.3235027,
            122960668.4278613,
            -4644650.528541619,
            218690.4374550451,
            -10150.68576333229,
            471.342195341046,
            -21.88655097124948,
            1.016166966590067,
            -0.04691224650074476,
            0.001590245644093093,
            0.00119268423306982,
            -0.002782929877162913,
        };
        auto result = SplineDerivCalculator(
            make_span(x), make_span(y), BC::not_a_knot)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);

        SplineInterpolator interpolate({x[0], y[0], result[0]},
                                       {x[1], y[1], result[1]});
        EXPECT_SOFT_EQ(834.53755181860117, interpolate(1.1e-4));
        EXPECT_SOFT_EQ(644.87140412068834, interpolate(5e-4));
    }
    {
        static double const expected_result[] = {
            122327823.66564,
            110798659.11594,
            -4492986.3810467,
            211979.81140063,
            -9839.4174867823,
            456.88890796125,
            -21.215420139232,
            0.98500714700094,
            -0.045473725878422,
            0.0015414822331671,
            0.0011561116748754,
            -0.0026975939080425,
        };
        auto result
            = SplineDerivCalculator(make_span(x), make_span(y), BC::geant)();
        EXPECT_VEC_SOFT_EQ(expected_result, result);

        SplineInterpolator interpolate({x[0], y[0], result[0]},
                                       {x[1], y[1], result[1]});
        EXPECT_SOFT_EQ(834.59530552174078, interpolate(1.1e-4));
        EXPECT_SOFT_EQ(646.15145372907716, interpolate(5e-4));
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
        = SplineDerivCalculator(make_span(x), make_span(y), BC::natural)();
    {
        SplineInterpolator interpolate({x[0], y[0], y_prime[0]},
                                       {x[1], y[1], y_prime[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.299, interpolate(0.1));
        EXPECT_SOFT_EQ(1.375, interpolate(0.5));
        EXPECT_SOFT_EQ(1.971, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));
    }
    y_prime
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
    y_prime = SplineDerivCalculator(make_span(x), make_span(y), BC::geant)();
    {
        SplineInterpolator interpolate({x[0], y[0], y_prime[0]},
                                       {x[1], y[1], y_prime[1]});
        EXPECT_EQ(0, interpolate(0));
        EXPECT_SOFT_EQ(0.32290625, interpolate(0.1));
        EXPECT_SOFT_EQ(1.26953125, interpolate(0.5));
        EXPECT_SOFT_EQ(1.87115625, interpolate(0.9));
        EXPECT_EQ(2, interpolate(1));
    }
    {
        SplineInterpolator interpolate({x[1], y[1], y_prime[1]},
                                       {x[2], y[2], y_prime[2]});
        EXPECT_EQ(2, interpolate(1));
        EXPECT_SOFT_EQ(1.23046875, interpolate(1.5));
        EXPECT_SOFT_EQ(0.99956465553125, interpolate(1.999));
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
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_EQ(60.498378344017667, calc_xs(Energy(99.99)));
    }
    else
    {
        EXPECT_SOFT_EQ(60.439491271972656, calc_xs(Energy(99.99)));
    }
    EXPECT_SOFT_EQ(100, calc_xs(Energy(100)));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
