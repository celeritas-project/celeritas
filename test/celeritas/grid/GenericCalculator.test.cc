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

    template<class T>
    using ItemRef = Collection<T, Ownership::const_reference, MemSpace::host>;

    void build(Span<real_type const> x, Span<real_type const> y)
    {
        GenericGridBuilder build_grid(&reals_);
        grid_ = build_grid(x, y);
        reals_ref_ = reals_;

        CELER_ENSURE(grid_);
        CELER_ENSURE(!reals_ref_.empty());
    }

    GenericGridRecord grid_;
    Items<real_type> reals_;
    ItemRef<real_type> reals_ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GenericCalculatorTest, nonmonotonic)
{
    static real_type const grid[] = {1.0, 2.0, 1e2, 1e4};
    static real_type const value[] = {4.0, 8.0, 8.0, 2.0};
    this->build(grid, value);
    GenericCalculator calc(grid_, reals_ref_);

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

TEST_F(GenericCalculatorTest, inverse)
{
    static real_type const grid[] = {0.5, 1.0, 2.0, 4.0};
    static real_type const value[] = {-1, 0, 1, 2};
    this->build(grid, value);

    auto calc = GenericCalculator::from_inverse(grid_, reals_ref_);
    EXPECT_SOFT_EQ(0.5, calc(-2));
    EXPECT_SOFT_EQ(0.5, calc(-1));
    EXPECT_SOFT_EQ(0.75, calc(-0.5));
    EXPECT_SOFT_EQ(3, calc(1.5));
    EXPECT_SOFT_EQ(4, calc(2));
    EXPECT_SOFT_EQ(4, calc(3));

    auto uninverted_calc = calc.make_inverse();
    for (auto x : grid)
    {
        EXPECT_SOFT_EQ(x, calc(uninverted_calc(x)));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
