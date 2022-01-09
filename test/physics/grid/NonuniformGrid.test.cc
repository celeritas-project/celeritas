//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file NonuniformGrid.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/NonuniformGrid.hh"

#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "celeritas_test.hh"

using celeritas::NonuniformGrid;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class NonuniformGridTest : public celeritas::Test
{
  protected:
    using GridT = NonuniformGrid<int>;

    void SetUp() override
    {
        auto build = celeritas::make_builder(&data);
        irange     = build.insert_back({0, 1, 3, 3, 7});
        ref        = data;
    }

    celeritas::ItemRange<int> irange;
    celeritas::Collection<int, celeritas::Ownership::value, celeritas::MemSpace::host>
        data;
    celeritas::Collection<int,
                          celeritas::Ownership::const_reference,
                          celeritas::MemSpace::host>
        ref;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

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
    EXPECT_THROW(grid.find(-1), celeritas::DebugError);
#endif
    EXPECT_EQ(0, grid.find(0));
    EXPECT_EQ(1, grid.find(1));
    EXPECT_EQ(1, grid.find(2));
    EXPECT_EQ(2, grid.find(3));
    EXPECT_EQ(3, grid.find(4));
#if CELERITAS_DEBUG
    EXPECT_THROW(grid.find(7), celeritas::DebugError);
    EXPECT_THROW(grid.find(10), celeritas::DebugError);
#endif
}
