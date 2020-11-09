//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UniformGrid.test.cc
//---------------------------------------------------------------------------//
#include "base/UniformGrid.hh"

#include "celeritas_test.hh"

using celeritas::UniformGrid;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformGridTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        params.size  = 3; //!< Number of grid points (2 bins!)
        params.front = 1.0;
        params.delta = 1.5;
    }

    celeritas::UniformGrid::Params params;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformGridTest, accessors)
{
    UniformGrid grid(params);
    EXPECT_EQ(3, grid.size());
    EXPECT_DOUBLE_EQ(1.0, grid.front());
    EXPECT_DOUBLE_EQ(1.0 + 1.5 * 2, grid.back());
    EXPECT_DOUBLE_EQ(1.0, grid[0]);
    EXPECT_DOUBLE_EQ(1.0 + 1.5 * 2, grid[2]);
}

TEST_F(UniformGridTest, find)
{
    UniformGrid grid(params);
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(0.99999), celeritas::DebugError);
#endif
    EXPECT_EQ(0, grid.find(1.0));
    EXPECT_EQ(0, grid.find(2.49999));
    EXPECT_EQ(1, grid.find(2.5));
    EXPECT_EQ(1, grid.find(3.99999));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(4.0), celeritas::DebugError);
    EXPECT_THROW(grid.find(4.0 + 0.00001), celeritas::DebugError);
#endif
}
