//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/DerivativeGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/DerivativeGridBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class DerivativeGridBuilderTest : public ::celeritas::test::Test
{
  protected:
    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Test derivative grid construction
TEST_F(DerivativeGridBuilderTest, build)
{
    real_type epsilon = 1e-8;

    inp::Grid grid;
    grid.x = {0.0, 0.4, 0.9, 1.3};
    grid.y = {-31.0, 12.1, 15.5, 92.0};

    DerivativeGridBuilder build(&scalars_, epsilon);
    NonuniformGridRecord grid_data = build(grid);

    EXPECT_TRUE(grid_data);
    EXPECT_EQ(16, scalars_.size());
    EXPECT_EQ(8, grid_data.grid.size());
    EXPECT_EQ(8, grid_data.grid.size());

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

    EXPECT_VEC_SOFT_EQ(expected_grid_x, scalars_[grid_data.grid]);
    EXPECT_VEC_SOFT_EQ(expected_grid_y, scalars_[grid_data.value]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
