//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/NonuniformGrid.test.cc
//---------------------------------------------------------------------------//

#include "corecel/grid/NonuniformGrid.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class NonuniformGridTest : public Test
{
  protected:
    using GridT = NonuniformGrid<int>;

    void SetUp() override
    {
        auto build = make_builder(&data);
        irange = build.insert_back({0, 1, 3, 3, 7});
        ref = data;
    }

    ItemRange<int> irange;
    Collection<int, Ownership::value, MemSpace::host> data;
    Collection<int, Ownership::const_reference, MemSpace::host> ref;
};

TEST_F(NonuniformGridTest, accessors)
{
    GridT grid(irange, ref);

    EXPECT_EQ(5, grid.size());
    EXPECT_EQ(0, grid.front());
    EXPECT_EQ(7, grid.back());
    EXPECT_EQ(0, grid[0]);
    EXPECT_EQ(3, grid[2]);
}

TEST_F(NonuniformGridTest, find)
{
    GridT grid(irange, ref);

#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(-1), DebugError);
#endif
    EXPECT_EQ(0, grid.find(0));
    EXPECT_EQ(1, grid.find(1));
    EXPECT_EQ(1, grid.find(2));
    EXPECT_EQ(2, grid.find(3));
    EXPECT_EQ(3, grid.find(4));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(7), DebugError);
    EXPECT_THROW(grid.find(10), DebugError);
#endif
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
