//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/DerivativeGridCalculator.hh"

#include "corecel/math/NumericLimits.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using DerivativeGridCalculatorTest = Test;

// Test derivative grid construction
TEST_F(DerivativeGridCalculatorTest, build)
{
    inp::Grid grid;
    grid.x = {0.0, 0.4, 0.9, 1.3};
    grid.y = {-31.0, 12.1, 15.5, 92.0};

    inp::Grid deriv_grid = construct_derivative_grid(grid);

    EXPECT_TRUE(deriv_grid);

    static real_type const expected_grid_x[] = {0, 0.2, 0.65, 1.3};

    static real_type const expected_grid_y[] = {107.75, 107.75, 6.8, 191.25};

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

    {
        // Coincident grid points are no longer supported
        // nowthat the derivative at each point has been
        // reduced to single harmonic mean value. The
        // derivative is undefined across a zero-width interval.
        EXPECT_THROW(construct_derivative_grid(grid), RuntimeError);
    }
}

//---------------------------------------------------------------------------//
// Test with near coincident points
TEST_F(DerivativeGridCalculatorTest, near_coincident)
{
    inp::Grid grid;
    grid.x = {0.0, 1.0, 1.001, 1.002, 1.003, 1.004, 2.0, 2.001, 3.0, 3.001};
    grid.y = {-10.0, 1.0, 2.0, 4.0, 5.0, 7.0, 10.0, 20.0, 100.0, 200.0};

    inp::Grid deriv_grid = construct_derivative_grid(grid);

    EXPECT_TRUE(deriv_grid);

    static real_type const expected_grid_y[] = {11,
                                                11,
                                                1000.00000000011,
                                                1999.99999999978,
                                                1000.00000000011,
                                                1999.99999999978,
                                                3.01204819277108,
                                                10000.0000000011,
                                                80.0800800800801,
                                                100000.00000001};

    EXPECT_VEC_SOFT_EQ(expected_grid_y, deriv_grid.y);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
