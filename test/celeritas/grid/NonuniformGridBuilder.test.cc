//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/NonuniformGridBuilder.hh"

#include <iostream>
#include <vector>

#include "celeritas/inp/Physics.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class NonuniformGridBuilderTest : public ::celeritas::test::Test
{
  protected:
    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(NonuniformGridBuilderTest, build)
{
    inp::Grid grid;
    grid.x = {0.0, 0.4, 0.9, 1.3};
    grid.y = {-31.0, 12.1, 15.5, 92.0};

    NonuniformGridBuilder build(&scalars_);
    NonuniformGridRecord grid_data = build(grid);

    EXPECT_TRUE(grid_data);
    EXPECT_EQ(8, scalars_.size());
    EXPECT_EQ(4, grid_data.grid.size());
    EXPECT_EQ(4, grid_data.value.size());

    EXPECT_VEC_SOFT_EQ(grid.x, scalars_[grid_data.grid]);
    EXPECT_VEC_SOFT_EQ(grid.y, scalars_[grid_data.value]);

    // Add another grid with the same x but different y
    // (This should deduplicate the x grid)
    grid.y = {19.0, 4.0, 6.0, 11.0};
    NonuniformGridRecord second_record = build(grid);
    EXPECT_EQ(grid_data.grid[0], second_record.grid[0]);
    EXPECT_EQ(grid_data.grid.size(), second_record.grid.size());
    EXPECT_NE(grid_data.value[0], second_record.value[0]);
    EXPECT_VEC_SOFT_EQ(grid.y, scalars_[second_record.value]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
