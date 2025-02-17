//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include <algorithm>
#include <cmath>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/grid/SplineDerivCalculator.hh"
#include "celeritas/grid/NonuniformGridBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class NonuniformGridCalculatorTest : public Test
{
  protected:
    using BC = SplineDerivCalculator::BoundaryCondition;
    using Values = Collection<real_type, Ownership::value, MemSpace::host>;
    using ValuesRef
        = Collection<real_type, Ownership::const_reference, MemSpace::host>;

    void build_spline(Span<real_type const> x, Span<real_type const> y, BC bc)
    {
        NonuniformGridBuilder build_grid(&reals_, bc);
        grid_ = build_grid(x, y);
        reals_ref_ = reals_;

        CELER_ENSURE(grid_);
        CELER_ENSURE(!grid_.derivative.empty() || bc == BC::size_);
        CELER_ENSURE(!reals_ref_.empty());
    }

    void build(Span<real_type const> x, Span<real_type const> y)
    {
        this->build_spline(x, y, BC::size_);
    }

    NonuniformGridRecord grid_;
    Values reals_;
    ValuesRef reals_ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(NonuniformGridCalculatorTest, nonmonotonic)
{
    static real_type const grid[] = {1.0, 2.0, 1e2, 1e4};
    static real_type const value[] = {4.0, 8.0, 8.0, 2.0};
    this->build(grid, value);
    NonuniformGridCalculator calc(grid_, reals_ref_);

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

TEST_F(NonuniformGridCalculatorTest, inverse)
{
    static real_type const grid[] = {0.5, 1.0, 2.0, 4.0};
    static real_type const value[] = {-1, 0, 1, 2};
    this->build(grid, value);

    auto calc = NonuniformGridCalculator::from_inverse(grid_, reals_ref_);
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

TEST_F(NonuniformGridCalculatorTest, spline)
{
    static real_type const grid[] = {0, 1, 3, 7, 9, 10};
    static real_type const value[] = {0, 1, 0, 1, 0, 1};
    this->build_spline(grid, value, BC::not_a_knot);

    auto calc = NonuniformGridCalculator(grid_, reals_ref_);
    EXPECT_SOFT_EQ(0, calc(0));
    EXPECT_SOFT_EQ(0.6184210526315791, calc(2));
    EXPECT_SOFT_EQ(-0.07360197368421052, calc(3.5));
    EXPECT_SOFT_EQ(1.073601973684211, calc(6.5));
    EXPECT_SOFT_EQ(1, calc(10));
    EXPECT_SOFT_EQ(1, calc(100));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
