//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GridInserter.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <vector>

#include "corecel/cont/Range.hh"
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
    Collection<XsGridRecord, Ownership::value, MemSpace::host> grids;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GridInserterTest, xs)
{
    // TODO: test uniform and nonuniform inserters
    XsGridInserter insert(&reals, &grids);
    {
        VecDbl values = {10, 20, 3};
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
        VecDbl values = {1, 2, 4, 6, 8};

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
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
