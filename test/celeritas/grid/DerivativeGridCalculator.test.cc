//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/DerivativeGridCalculator.test.cc
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
    real_type epsilon = 1e-8;

    inp::Grid grid;
    grid.x = {0.0, 0.4, 0.9, 1.3};
    grid.y = {-31.0, 12.1, 15.5, 92.0};

    DerivativeGridCalculator build(epsilon);
    inp::Grid deriv_grid = build(grid);

    EXPECT_TRUE(deriv_grid);
    EXPECT_EQ(8, deriv_grid.x.size());
    EXPECT_EQ(8, deriv_grid.y.size());

    static real_type const expected_grid_x[] = {
        0 - epsilon,
        0 + epsilon,
        0.4 - epsilon,
        0.4 + epsilon,
        0.9 - epsilon,
        0.9 + epsilon,
        1.3 - epsilon,
        1.3 + epsilon,
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
}  // namespace test
}  // namespace celeritas
