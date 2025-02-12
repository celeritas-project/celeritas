//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridInserter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/ValueGridInserter.hh"

#include <algorithm>
#include <vector>

#include "corecel/cont/Range.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ValueGridInserterTest : public Test
{
  protected:
    Collection<real_type, Ownership::value, MemSpace::host> real_storage;
    Collection<XsGridData, Ownership::value, MemSpace::host> grid_storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridInserterTest, all)
{
    ValueGridInserter insert(&real_storage, &grid_storage);

    {
        std::vector<double> const values = {10, 20, 3};

        auto idx = insert(
            UniformGridData::from_bounds(0.0, 1.0, 3), 1, make_span(values));
        EXPECT_EQ(0, idx.unchecked_get());
        XsGridData const& inserted = grid_storage[idx];

        EXPECT_EQ(3, inserted.log_energy.size);
        EXPECT_EQ(1, inserted.prime_index);
        EXPECT_VEC_SOFT_EQ(values, real_storage[inserted.value]);
    }
    {
        std::vector<double> const values = {1, 2, 4, 6, 8};

        auto idx = insert(UniformGridData::from_bounds(0.0, 10.0, 5),
                          make_span(values));
        EXPECT_EQ(1, idx.unchecked_get());
        XsGridData const& inserted = grid_storage[idx];

        EXPECT_EQ(5, inserted.log_energy.size);
        EXPECT_EQ(XsGridData::no_scaling(), inserted.prime_index);
        EXPECT_VEC_SOFT_EQ(values, real_storage[inserted.value]);
    }
    EXPECT_EQ(2, grid_storage.size());
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
