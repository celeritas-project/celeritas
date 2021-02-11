//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridInserter.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/ValueGridInserter.hh"

#include <algorithm>
#include "celeritas_test.hh"
#include "base/Range.hh"

using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ValueGridInserterTest : public celeritas::Test
{
  protected:
    Pie<real_type, Ownership::value, MemSpace::host>  real_storage;
    Pie<XsGridData, Ownership::value, MemSpace::host> grid_storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridInserterTest, all)
{
    ValueGridInserter insert(&real_storage, &grid_storage);

    {
        const real_type values[] = {10, 20, 3};

        auto idx = insert(
            UniformGridData::from_bounds(0.0, 1.0, 3), 1, make_span(values));
        EXPECT_EQ(0, idx.unchecked_get());
        const XsGridData& inserted = grid_storage[idx];

        EXPECT_EQ(3, inserted.log_energy.size);
        EXPECT_EQ(1, inserted.prime_index);
        EXPECT_VEC_SOFT_EQ(values, real_storage[inserted.value]);
    }
    {
        const real_type values[] = {1, 2, 4, 6, 8};

        auto idx = insert(UniformGridData::from_bounds(0.0, 10.0, 5),
                          make_span(values));
        EXPECT_EQ(1, idx.unchecked_get());
        const XsGridData& inserted = grid_storage[idx];

        EXPECT_EQ(5, inserted.log_energy.size);
        EXPECT_EQ(XsGridData::no_scaling(), inserted.prime_index);
        EXPECT_VEC_SOFT_EQ(values, real_storage[inserted.value]);
    }
    EXPECT_EQ(2, grid_storage.size());
}
