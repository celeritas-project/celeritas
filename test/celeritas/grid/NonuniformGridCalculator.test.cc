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
#include "corecel/inp/Grid.hh"
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

    void build(inp::Grid const& grid)
    {
        NonuniformGridBuilder build_grid(&reals_);
        grid_ = build_grid(grid);
        reals_ref_ = reals_;

        CELER_ENSURE(grid_);
        CELER_ENSURE(!grid_.derivative.empty()
                     || grid.interpolation.type
                            != InterpolationType::cubic_spline);
        CELER_ENSURE(!reals_ref_.empty());
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
    inp::Grid grid;
    grid.x = {1.0, 2.0, 1e2, 1e4};
    grid.y = {4.0, 8.0, 8.0, 2.0};
    this->build(grid);

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

TEST_F(NonuniformGridCalculatorTest, discontinuous)
{
    inp::Grid grid;
    grid.x = {1.0, 2.0, 2.0, 3.0};
    grid.y = {1.0, 1.0, 2.0, 2.0};
    this->build(grid);

    NonuniformGridCalculator calc(grid_, reals_ref_);

    // Test accessing tabulated data
    EXPECT_EQ(1.0, calc[0]);
    EXPECT_EQ(2.0, calc[3]);

    // Test on grid points
    EXPECT_SOFT_EQ(1.0, calc(1.0));
    EXPECT_SOFT_EQ(2.0, calc(2.0));
    EXPECT_SOFT_EQ(2.0, calc(3.0));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(1.0, calc(0.0));
    EXPECT_SOFT_EQ(2.0, calc(4.0));
}

TEST_F(NonuniformGridCalculatorTest, discontinuous_end)
{
    inp::Grid grid;
    grid.x = {1.0, 2.0, 2.0};
    grid.y = {1.0, 1.0, 2.0};
    this->build(grid);

    NonuniformGridCalculator calc(grid_, reals_ref_);

    // Test accessing tabulated data
    EXPECT_EQ(1.0, calc[0]);
    EXPECT_EQ(2.0, calc[2]);

    // Test on grid points
    EXPECT_SOFT_EQ(1.0, calc(1));
    EXPECT_SOFT_EQ(2.0, calc(2));

    // Test out-of-bounds
    EXPECT_SOFT_EQ(1.0, calc(0.0));
    EXPECT_SOFT_EQ(2.0, calc(3.0));
}
TEST_F(NonuniformGridCalculatorTest, discontinuous_all)
{
    inp::Grid grid;
    grid.x = {2.0, 2.0};
    grid.y = {-1.0, 1.0};
    this->build(grid);

    NonuniformGridCalculator calc(grid_, reals_ref_);

    // Test on and around the single coincident point
    EXPECT_SOFT_EQ(-1.0, calc(1.9));
    EXPECT_SOFT_EQ(1.0, calc(2.0));
    EXPECT_SOFT_EQ(1.0, calc(2.1));
}

TEST_F(NonuniformGridCalculatorTest, inverse)
{
    inp::Grid grid;
    grid.x = {0.5, 1.0, 2.0, 4.0};
    grid.y = {-1, 0, 1, 2};
    this->build(grid);

    auto calc = NonuniformGridCalculator::from_inverse(grid_, reals_ref_);

    EXPECT_SOFT_EQ(0.5, calc(-2));
    EXPECT_SOFT_EQ(0.5, calc(-1));
    EXPECT_SOFT_EQ(0.75, calc(-0.5));
    EXPECT_SOFT_EQ(3, calc(1.5));
    EXPECT_SOFT_EQ(4, calc(2));
    EXPECT_SOFT_EQ(4, calc(3));

    auto uninverted_calc = calc.make_inverse();
    for (auto x : grid.x)
    {
        EXPECT_SOFT_EQ(x, calc(uninverted_calc(x)));
    }
}

TEST_F(NonuniformGridCalculatorTest, spline)
{
    inp::Grid grid;
    grid.x = {0, 1, 3, 7, 9, 10};
    grid.y = {0, 1, 0, 1, 0, 1};
    grid.interpolation.type = InterpolationType::cubic_spline;
    grid.interpolation.bc = BC::not_a_knot;
    this->build(grid);

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
