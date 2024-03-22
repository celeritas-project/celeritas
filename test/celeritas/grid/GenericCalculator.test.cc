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
#include "corecel/data/Ref.hh"
#include "celeritas/grid/GenericGridBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GenericCalculatorTest : public Test
{
  protected:
    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;

    void SetUp() override
    {
        std::vector<real_type> const grid = {1.0, 2.0, 1e2, 1e4};
        std::vector<real_type> const value = {4.0, 8.0, 8.0, 2.0};

        GenericGridBuilder build_grid(&reals_);
        grid_ = build_grid(make_span(grid), make_span(value));
        CELER_ENSURE(grid_);
    }

    GenericGridData grid_;
    Items<real_type> reals_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GenericCalculatorTest, all)
{
    Collection<real_type, Ownership::const_reference, MemSpace::host> ref;
    ref = reals_;
    GenericCalculator calc(grid_, ref);

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
