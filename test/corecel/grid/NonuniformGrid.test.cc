//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

    void SetUp() override { this->build({-1, 1, 1, 3, 3, 3, 8}); }

    void build(std::initializer_list<int> grid)
    {
        auto build = CollectionBuilder{&data};
        irange = build.insert_back(grid);
        ref = data;
    }

    ItemRange<int> irange;
    Collection<int, Ownership::value, MemSpace::host> data;
    Collection<int, Ownership::const_reference, MemSpace::host> ref;
};

TEST_F(NonuniformGridTest, accessors)
{
    GridT grid(irange, ref);

    EXPECT_EQ(7, grid.size());
    EXPECT_EQ(-1, grid.front());
    EXPECT_EQ(8, grid.back());
    EXPECT_EQ(-1, grid[0]);
    EXPECT_EQ(3, grid[3]);
}

TEST_F(NonuniformGridTest, find)
{
    GridT grid(irange, ref);

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(grid.find(-2), DebugError);
    }
    else
    {
        EXPECT_EQ(0, grid.find(-2));
    }
    EXPECT_EQ(0, grid.find(-1));
    EXPECT_EQ(0, grid.find(0));
    EXPECT_EQ(2, grid.find(1));
    EXPECT_EQ(2, grid.find(2));
    EXPECT_EQ(5, grid.find(3));
    EXPECT_EQ(5, grid.find(4));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(grid.find(8), DebugError);
        EXPECT_THROW(grid.find(10), DebugError);
    }
    else
    {
        EXPECT_EQ(6, grid.find(10));
        EXPECT_EQ(6, grid.find(10));
    }
}

TEST_F(NonuniformGridTest, values)
{
    GridT grid(irange, ref);

    auto values = grid.values();
    ASSERT_EQ(grid.size(), values.size());

    for (auto i : range(grid.size()))
    {
        EXPECT_EQ(grid[i], values[i]);
    }
}

TEST_F(NonuniformGridTest, degenerate)
{
    // Single-point grids are not allowed
    if (CELERITAS_DEBUG)
    {
        SCOPED_TRACE("single point");
        this->build({1});
        EXPECT_THROW(GridT(irange, ref), DebugError);
    }

    {
        SCOPED_TRACE("two coincident");
        this->build({1, 1});
        GridT grid(irange, ref);
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(grid.find(1), DebugError);
        }
        else
        {
            EXPECT_EQ(1, grid.find(1));
        }
    }
    {
        SCOPED_TRACE("three coincident");
        this->build({1, 1, 1});
        GridT grid(irange, ref);
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(grid.find(1), DebugError);
        }
        else
        {
            EXPECT_EQ(2, grid.find(1));
        }
    }
    {
        SCOPED_TRACE("front coincident");
        this->build({1, 1, 3});
        GridT grid(irange, ref);
        EXPECT_EQ(1, grid.find(1));
        EXPECT_EQ(1, grid.find(2));
    }
    {
        SCOPED_TRACE("back coincident");
        this->build({-1, 1, 1});
        GridT grid(irange, ref);
        EXPECT_EQ(0, grid.find(0));
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(grid.find(1), DebugError);
            EXPECT_THROW(grid.find(2), DebugError);
        }
        else
        {
            EXPECT_EQ(2, grid.find(1));
            EXPECT_EQ(2, grid.find(2));
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
