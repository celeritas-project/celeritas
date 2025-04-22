//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/UniformGrid.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/UniformGrid.hh"

#include <cmath>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class UniformGridTest : public Test
{
  protected:
    void SetUp() override
    {
        input.size = 3;  //!< Number of grid points (2 bins!)
        input.front = 1.0;
        input.delta = 1.5;
        input.back = input.front + input.delta * (input.size - 1);
    }

    UniformGridData input;
};

TEST_F(UniformGridTest, accessors)
{
    ASSERT_TRUE(input);

    UniformGrid grid(input);
    EXPECT_EQ(3, grid.size());
    EXPECT_REAL_EQ(1.0, grid.front());
    EXPECT_REAL_EQ(1.0 + 1.5 * 2, grid.back());
    EXPECT_REAL_EQ(1.0, grid[0]);
    EXPECT_REAL_EQ(1.0 + 1.5 * 2, grid[2]);
}

TEST_F(UniformGridTest, find)
{
    ASSERT_TRUE(input);

    UniformGrid grid(input);
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(0.99999), DebugError);
#endif
    EXPECT_EQ(0, grid.find(1.0));
    EXPECT_EQ(0, grid.find(2.49999));
    EXPECT_EQ(1, grid.find(2.5));
    EXPECT_EQ(1, grid.find(3.99999));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(4.0), DebugError);
    EXPECT_THROW(grid.find(4.0 + 0.00001), DebugError);
#endif
}

TEST_F(UniformGridTest, from_bounds)
{
    input = UniformGridData::from_bounds({-1, 5}, 7);
    ASSERT_TRUE(input);

    UniformGrid grid(input);
    EXPECT_EQ(7, grid.size());
    EXPECT_REAL_EQ(-1.0, grid.front());
    EXPECT_REAL_EQ(5.0, grid.back());
    EXPECT_EQ(1, grid.find(0.0));
}

TEST_F(UniformGridTest, TEST_IF_CELERITAS_DOUBLE(from_logbounds))
{
    real_type const log_emin = std::log(real_type{1.0});
    real_type const log_emax = std::log(real_type{1e5});
    input = UniformGridData::from_bounds({log_emin, log_emax}, 6);

    UniformGrid grid(input);
    EXPECT_EQ(6, grid.size());
    EXPECT_EQ(log_emin, grid.front());
    EXPECT_REAL_EQ(log_emax, grid.back());
    EXPECT_EQ(0, grid.find(log_emin));

    real_type const ten{10};
    real_type const log10 = std::log(ten);
    EXPECT_EQ(0, grid.find(std::nextafter(log10, 0.)));
    EXPECT_EQ(1, grid.find(std::log(ten)));
    EXPECT_EQ(1, grid.find(std::nextafter(log10, 1e100)));
    EXPECT_EQ(4, grid.find(std::nextafter(log_emax, 0.)));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(log_emax), DebugError);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
