//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GridInserter.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <vector>

#include "corecel/cont/Range.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/grid/UniformGridInserter.hh"
#include "celeritas/grid/XsGridInserter.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GridInserterTest : public Test
{
  protected:
    using VecDbl = std::vector<double>;
    Collection<real_type, Ownership::value, MemSpace::host> reals;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GridInserterTest, xs)
{
    Collection<XsGridRecord, Ownership::value, MemSpace::host> grids;
    XsGridInserter insert(&reals, &grids);
    {
        VecDbl const values = {10, 20, 3};
        auto lower = UniformGridData::from_bounds(1e-2, 1e-1, 2);
        auto upper = UniformGridData::from_bounds(1e-1, 1, 2);

        auto idx = insert(lower,
                          make_span(values).subspan(0, 2),
                          upper,
                          make_span(values).subspan(1, 2));
        EXPECT_EQ(0, idx.unchecked_get());
        XsGridRecord const& inserted = grids[idx];

        EXPECT_TRUE(inserted.lower);
        EXPECT_TRUE(inserted.upper);
        EXPECT_EQ(2, inserted.lower.grid.size);
        EXPECT_EQ(2, inserted.upper.grid.size);
        EXPECT_VEC_SOFT_EQ(make_span(values).subspan(0, 2),
                           reals[inserted.lower.value]);
    }
    {
        VecDbl const values = {1, 2, 4, 6, 8};

        auto idx = insert(UniformGridData::from_bounds(0.0, 10.0, 5),
                          make_span(values));
        EXPECT_EQ(1, idx.unchecked_get());
        XsGridRecord const& inserted = grids[idx];

        EXPECT_TRUE(inserted.lower);
        EXPECT_FALSE(inserted.upper);
        EXPECT_EQ(5, inserted.lower.grid.size);
        EXPECT_VEC_SOFT_EQ(values, reals[inserted.lower.value]);
    }
    EXPECT_EQ(2, grids.size());
}

TEST_F(GridInserterTest, uniform)
{
    Collection<UniformGridRecord, Ownership::value, MemSpace::host> grids;

    UniformGridInserter insert(&reals, &grids);
    VecDbl const values = {1, 2, 4, 6, 8};

    auto idx = insert(UniformGridData::from_bounds(0.0, 10.0, 5),
                      make_span(values));
    EXPECT_EQ(0, idx.unchecked_get());
    UniformGridRecord const& inserted = grids[idx];
    EXPECT_EQ(1, grids.size());

    EXPECT_TRUE(inserted);
    EXPECT_EQ(5, inserted.grid.size);
    EXPECT_EQ(0, inserted.grid.front);
    EXPECT_EQ(10, inserted.grid.back);
    EXPECT_VEC_SOFT_EQ(values, reals[inserted.value]);
}

TEST_F(GridInserterTest, nonuniform)
{
    Collection<NonuniformGridRecord, Ownership::value, MemSpace::host> grids;

    NonuniformGridInserter insert(&reals, &grids);
    VecDbl const grid = {0, 1, 2, 5, 13};
    VecDbl const values = {1, 2, 4, 6, 8};

    auto idx = insert(make_span(grid), make_span(values));
    EXPECT_EQ(0, idx.unchecked_get());
    NonuniformGridRecord const& inserted = grids[idx];
    EXPECT_EQ(1, grids.size());

    EXPECT_TRUE(inserted);
    EXPECT_VEC_SOFT_EQ(grid, reals[inserted.grid]);
    EXPECT_VEC_SOFT_EQ(values, reals[inserted.value]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
