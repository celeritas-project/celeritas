//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/DerivativeGridCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DerivativeGridCalculatorTest : public ::celeritas::test::Test
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test derivative grid construction
TEST_F(DerivativeGridCalculatorTest, build)
{
    inp::Grid grid;
    grid.x = {0.0, 0.4, 0.9, 1.3};
    grid.y = {-31.0, 12.1, 15.5, 92.0};

    inp::Grid deriv_grid = construct_derivative_grid(grid);

    EXPECT_TRUE(deriv_grid);

    static real_type const expected_grid_x[] = {
        0,
        0,
        0.4,
        0.4,
        0.9,
        0.9,
        1.3,
        1.3,
    };

    static real_type const expected_grid_y[] = {
        0,
        107.75,
        107.75,
        6.8,
        6.8,
        191.25,
        191.25,
        0,
    };

    EXPECT_VEC_SOFT_EQ(expected_grid_x, deriv_grid.x);
    EXPECT_VEC_SOFT_EQ(expected_grid_y, deriv_grid.y);
}

//---------------------------------------------------------------------------//
// Test with coincident points
TEST_F(DerivativeGridCalculatorTest, coincident)
{
    inp::Grid grid;
    grid.x = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
    grid.y = {-10.0, 1.0, 2.0, 4.0, 5.0, 7.0, 10.0, 20.0, 100.0, 200.0};

    inp::Grid deriv_grid = construct_derivative_grid(grid);

    EXPECT_TRUE(deriv_grid);

    real_type inf = NumericLimits<real_type>::infinity();

    static real_type const expected_grid_x[] = {
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
    };

    static real_type const expected_grid_y[] = {
        0,   11.0, 11.0, inf, inf, inf,  inf,  inf, inf, inf,
        inf, 3.0,  3.0,  inf, inf, 80.0, 80.0, inf, inf, 0,
    };

    EXPECT_VEC_SOFT_EQ(expected_grid_x, deriv_grid.x);
    EXPECT_VEC_SOFT_EQ(expected_grid_y, deriv_grid.y);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
