//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/FindInterp.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/FindInterp.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/grid/UniformGrid.hh"
#include "corecel/math/Turn.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(FindInterpTest, uniform_real)
{
    auto data = UniformGridData::from_bounds({1.0, 5.0}, 3);
    UniformGrid const grid(data);

    {
        auto interp = find_interp(grid, 1.0);
        EXPECT_EQ(0, interp.index);
        EXPECT_SOFT_EQ(0.0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 3.0);
        EXPECT_EQ(1, interp.index);
        EXPECT_SOFT_EQ(0.0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 4.0);
        EXPECT_EQ(1, interp.index);
        EXPECT_SOFT_EQ(0.5, interp.fraction);
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(find_interp(grid, 0.999), DebugError);
        EXPECT_THROW(find_interp(grid, 5.0), DebugError);
        EXPECT_THROW(find_interp(grid, 5.001), DebugError);
    }
}

TEST(FindInterpTest, nonuniform)
{
    Collection<double, Ownership::value, MemSpace::host> data;
    auto build = CollectionBuilder{&data};
    auto const irange = build.insert_back({-2.0, -1.5, 1.5, 2.0, 2.0, 8.0});
    Collection<double, Ownership::const_reference, MemSpace::host> ref{data};
    NonuniformGrid<double> grid(irange, ref);

    {
        auto interp = find_interp(grid, -2);
        EXPECT_EQ(0, interp.index);
        EXPECT_DOUBLE_EQ(0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 0.0);
        EXPECT_EQ(1, interp.index);
        EXPECT_DOUBLE_EQ(.5, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 2.0);
        EXPECT_EQ(3, interp.index);
        EXPECT_TRUE(std::isnan(interp.fraction)) << repr(interp.fraction);
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(find_interp(grid, -3), DebugError);
        EXPECT_THROW(find_interp(grid, 8), DebugError);
    }
}

// In this case, the fraction is always truncated by integer division to zero
// If we actually care about this in the future we can return a rational number
// for the "value"
TEST(FindInterpTest, nonuniform_int)
{
    Collection<int, Ownership::value, MemSpace::host> data;
    auto build = CollectionBuilder{&data};
    auto const irange = build.insert_back({0, 2, 6, 6, 8});
    Collection<int, Ownership::const_reference, MemSpace::host> ref{data};
    NonuniformGrid<int> grid(irange, ref);

    {
        auto interp = find_interp(grid, 0);
        EXPECT_EQ(0, interp.index);
        EXPECT_EQ(0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 4);
        EXPECT_EQ(1, interp.index);
        EXPECT_EQ(0, interp.fraction);
    }
    if (false)
    {
        // Disabled: results in UB due to integer division
        auto interp = find_interp(grid, 6);
        EXPECT_EQ(2, interp.index);
        EXPECT_EQ(0, interp.fraction);
    }
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(find_interp(grid, -1), DebugError);
        EXPECT_THROW(find_interp(grid, 8), DebugError);
    }
}

TEST(FindInterpTest, Quantity)
{
    Collection<Turn, Ownership::value, MemSpace::host> data;
    auto build = CollectionBuilder{&data};
    auto const irange
        = build.insert_back({Turn{0}, Turn{0.5}, Turn{0.75}, Turn{1}});
    Collection<Turn, Ownership::const_reference, MemSpace::host> ref{data};
    NonuniformGrid<Turn> grid(irange, ref);

    {
        auto interp = find_interp(grid, Turn{0});
        EXPECT_EQ(0, interp.index);
        EXPECT_EQ(0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, Turn{0.625});
        EXPECT_EQ(1, interp.index);
        EXPECT_TRUE((
            std::is_same<decltype(interp.fraction), Turn::value_type>::value));
        EXPECT_EQ(0.5, interp.fraction);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
