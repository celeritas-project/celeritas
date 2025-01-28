//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineDerivativeCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/SplineDerivativeCalculator.hh"

#include <cmath>
#include <vector>

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

class SplineDerivativeCalculatorTest : public CalculatorTestBase
{
  protected:
    using VecReal = std::vector<real_type>;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SplineDerivativeCalculatorTest, simple)
{
    VecReal x{0, 1, 2, 3, 4};
    VecReal y{0, 2, 1, 2, 0};

    auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(VecReal({6, -3 / 4.0, 0, 3 / 4.0, -6}), result);
}

TEST_F(SplineDerivativeCalculatorTest, small)
{
    VecReal x{1, 2, 4};
    VecReal y{2, 4, 2};

    auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(VecReal({3, 1, -3}), result);
}

TEST_F(SplineDerivativeCalculatorTest, constant)
{
    VecReal x{0, 1, 3, 7};
    VecReal y{3, 3, 3, 3};

    auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();
    EXPECT_VEC_SOFT_EQ(VecReal({0, 0, 0, 0}), result);
}

TEST_F(SplineDerivativeCalculatorTest, sin)
{
    size_type num_points = 10;
    VecReal x(num_points);
    VecReal y(num_points);

    for (size_type i = 0; i < num_points; ++i)
    {
        x[i] = i;
        y[i] = std::sin(i);
    }

    auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();

    static double const expected_result[] = {
        1.1445931049699,
        0.49649878902935,
        -0.40269598061028,
        -0.9867677968323,
        -0.64853259846134,
        0.28076534250866,
        0.95763221975372,
        0.73643839862225,
        -0.09706457977579,
        -1.0827844199502,
    };
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(SplineDerivativeCalculatorTest, nonuniform)
{
    VecReal x{0, 7, 16, 20, 24, 25, 29, 31, 38, 44, 53, 55, 60, 67, 74, 81};
    VecReal y{7, 3, 4, 2, 1, 1, 5, 3, 3, 2, 7, 1, 8, 7, 8, 2};

    auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();

    static double const expected_result[] = {
        -1.5134392539008,
        0.089094798573181,
        -0.33218833778178,
        -0.43179434207034,
        -0.19063429393686,
        0.39703432035974,
        -0.20780602784999,
        -1.0750990766299,
        -0.096787212855884,
        0.78100885629034,
        -2.8431967955012,
        -1.8531505420245,
        1.7800457829244,
        -0.49838042777432,
        0.21347592817287,
        -2.4983804277743,
    };
    EXPECT_VEC_SOFT_EQ(expected_result, result);
}

TEST_F(SplineDerivativeCalculatorTest, xs_grid)
{
    this->build({0.01, 100}, 5, [](real_type energy) {
        return energy >= 1 ? energy : 1 / energy;
    });
    this->convert_to_prime(3);

    static double const expected_result[]
        = {-5697 / 5.0, -9516 / 11.0, 396, -3939 / 11.0, 3951};
    {
        auto result
            = SplineDerivativeCalculator(this->data(), this->values())();
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
        auto result = SplineDerivativeCalculator(make_span(x), make_span(y))();
        EXPECT_VEC_SOFT_EQ(expected_result, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
