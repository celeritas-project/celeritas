//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformLogGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/UniformLogGridCalculator.hh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "corecel/io/Repr.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformLogGridCalculatorTest : public CalculatorTestBase
{
  protected:
    using Energy = UniformLogGridCalculator::Energy;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformLogGridCalculatorTest, simple)
{
    // Energy from 1 to 1e5 MeV with 6 grid points; XS = E
    inp::UniformGrid grid;
    grid.x = {1, 1e5};
    grid.y = {1, 10, 1e2, 1e3, 1e4, 1e5};
    this->build(grid);

    UniformLogGridCalculator calc(this->uniform_grid(), this->values());

    // Test on grid points
    EXPECT_SOFT_EQ(1.0, calc(Energy{1}));
    EXPECT_SOFT_EQ(1e2, calc(Energy{1e2}));
    EXPECT_SOFT_EQ(1e5 - 1e-6, calc(Energy{1e5 - 1e-6}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e5}));

    // Test access by index
    EXPECT_SOFT_EQ(1.0, calc[0]);
    EXPECT_SOFT_EQ(1e2, calc[2]);
    EXPECT_SOFT_EQ(1e5, calc[5]);

    // Test between grid points
    EXPECT_SOFT_EQ(5, calc(Energy{5}));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(1.0, calc(Energy{0.0001}));
    EXPECT_SOFT_EQ(1e5, calc(Energy{1e7}));

    // Test energy grid bounds
    EXPECT_SOFT_EQ(1.0, value_as<Energy>(calc.energy_min()));
    EXPECT_SOFT_EQ(1e5, value_as<Energy>(calc.energy_max()));
}

TEST_F(UniformLogGridCalculatorTest, spline)
{
    inp::UniformGrid grid;
    grid.x = {1e-2, 1e2};
    grid.y = {100, 10, 1, 10, 100};
    grid.interpolation.type = InterpolationType::cubic_spline;
    grid.interpolation.bc = BC::not_a_knot;
    this->build(grid);

    UniformLogGridCalculator calc(this->uniform_grid(), this->values());
    EXPECT_SOFT_EQ(10, calc(Energy(0.1)));
    EXPECT_SOFT_EQ(-62.572615039281715, calc(Energy(0.2)));
    EXPECT_SOFT_EQ(1, calc(Energy(1)));
    EXPECT_SOFT_EQ(847.3120089786757, calc(Energy(5)));
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_EQ(60.498378344017667, calc(Energy(99.99)));
    }
    else
    {
        EXPECT_SOFT_EQ(60.439491271972656, calc(Energy(99.99)));
    }
    EXPECT_SOFT_EQ(100, calc(Energy(100)));
}

TEST_F(UniformLogGridCalculatorTest, spline_deriv)
{
    inp::UniformGrid grid;
    grid.x = {1e-2, 1e2};
    grid.y = {100, 10, 1, 10, 100};
    grid.interpolation.type = InterpolationType::cubic_spline;
    grid.interpolation.bc = BC::not_a_knot;
    this->build(grid);

    static double const expected_deriv[] = {
        105520 / 33.0, 31880 / 11.0, -3160 / 33.0, -790 / 11.0, 5530 / 33.0};

    SplineDerivCalculator calc_deriv(BC::not_a_knot);
    auto const& data = this->uniform_grid();
    {
        auto deriv = calc_deriv(data, this->values());
        EXPECT_VEC_SOFT_EQ(expected_deriv, deriv);
    }
    {
        std::vector<real_type> x{0.01, 0.1, 1, 10, 100};
        std::vector<real_type> y{100, 10, 1, 10, 100};

        UniformGrid loge_grid(data.grid);
        UniformLogGridCalculator calc(data, this->values());
        for (auto i : range(loge_grid.size()))
        {
            EXPECT_SOFT_EQ(x[i], std::exp(loge_grid[i]));
            EXPECT_SOFT_EQ(y[i], calc[i]);
        }
        auto deriv = calc_deriv(make_span(x), make_span(y));
        EXPECT_VEC_SOFT_EQ(expected_deriv, deriv);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
