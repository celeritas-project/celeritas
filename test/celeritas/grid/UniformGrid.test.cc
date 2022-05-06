//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/UniformGrid.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/UniformGrid.hh"

#include <cmath>

#include "celeritas_test.hh"

using celeritas::UniformGrid;
using celeritas::UniformGridData;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UniformGridTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        input.size  = 3; //!< Number of grid points (2 bins!)
        input.front = 1.0;
        input.delta = 1.5;
        input.back  = input.front + input.delta * (input.size - 1);
    }

    UniformGridData input;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UniformGridTest, accessors)
{
    ASSERT_TRUE(input);

    UniformGrid grid(input);
    EXPECT_EQ(3, grid.size());
    EXPECT_DOUBLE_EQ(1.0, grid.front());
    EXPECT_DOUBLE_EQ(1.0 + 1.5 * 2, grid.back());
    EXPECT_DOUBLE_EQ(1.0, grid[0]);
    EXPECT_DOUBLE_EQ(1.0 + 1.5 * 2, grid[2]);
}

TEST_F(UniformGridTest, find)
{
    ASSERT_TRUE(input);

    UniformGrid grid(input);
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

TEST_F(UniformGridTest, from_bounds)
{
    input = UniformGridData::from_bounds(-1, 5, 7);
    ASSERT_TRUE(input);

    UniformGrid grid(input);
    EXPECT_EQ(7, grid.size());
    EXPECT_DOUBLE_EQ(-1.0, grid.front());
    EXPECT_DOUBLE_EQ(5.0, grid.back());
    EXPECT_EQ(1, grid.find(0.0));
}

TEST_F(UniformGridTest, from_logbounds)
{
    const double log_emin = std::log(1.0);
    const double log_emax = std::log(1e5);
    input = UniformGridData::from_bounds(log_emin, log_emax, 6);

    UniformGrid grid(input);
    EXPECT_EQ(6, grid.size());
    EXPECT_EQ(log_emin, grid.front());
    EXPECT_DOUBLE_EQ(log_emax, grid.back());
    EXPECT_EQ(0, grid.find(log_emin));

    const double log10 = std::log(10);
    EXPECT_EQ(0, grid.find(std::nextafter(log10, 0.)));
    EXPECT_EQ(1, grid.find(std::log(10)));
    EXPECT_EQ(1, grid.find(std::nextafter(log10, 1e100)));
    EXPECT_EQ(4, grid.find(std::nextafter(log_emax, 0.)));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(log_emax), celeritas::DebugError);
#endif
}
