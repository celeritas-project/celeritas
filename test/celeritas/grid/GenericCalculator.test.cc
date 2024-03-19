//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/GenericCalculator.hh"

#include <algorithm>
#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"

#include "CalculatorTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GenericCalculatorTest : public CalculatorTestBase
{
  protected:
    void SetUp() override
    {
        std::vector<real_type> grid{1.0, 2.0, 1e2, 1e4};
        std::vector<real_type> value{4.0, 8.0, 8.0, 2.0};

        storage_ = {};
        data_.grid
            = make_builder(&storage_).insert_back(grid.begin(), grid.end());
        data_.value
            = make_builder(&storage_).insert_back(value.begin(), value.end());
        ref_ = storage_;

        CELER_ENSURE(data_);
    }

    GenericGridData data_;
    Values storage_;
    Data ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GenericCalculatorTest, all)
{
    GenericCalculator calc(data_, ref_);

    // Test accessing tabulated data
    EXPECT_EQ(4.0, calc[0]);
    EXPECT_EQ(2.0, calc[3]);

    // Test on grid points
    EXPECT_SOFT_EQ(4.0, calc(1));
    EXPECT_SOFT_EQ(8.0, calc(2));
    EXPECT_SOFT_EQ(8.0, calc(1e2));
    EXPECT_SOFT_EQ(2.0, calc(1e4));

    // Test between grid points
    EXPECT_SOFT_EQ(6.0, calc(1.5));
    EXPECT_SOFT_EQ(5.0, calc(5050));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(4.0, calc(0.0001));
    EXPECT_SOFT_EQ(2.0, calc(1e7));
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
