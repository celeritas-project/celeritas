//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/SplineCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/SplineCalculator.hh"

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

class SplineCalculatorTest : public CalculatorTestBase
{
  protected:
    using Energy = SplineCalculator::Energy;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SplineCalculatorTest, simple)
{
    // Energy from 1 to 1e5 MeV with 5 grid points; XS should be the same
    // *No* magical 1/E scaling
    inp::UniformGrid grid;
    grid.x = {1, 1e5};
    grid.y = {1, 10, 1e2, 1e3, 1e4, 1e5};

    for (size_type order = 1; order < 5; ++order)
    {
        grid.interpolation.order = order;
        this->build(grid);

        SplineCalculator calc(this->data().lower, this->values());

        // Test on grid points
        EXPECT_SOFT_EQ(1.0, calc(Energy{1}));
        EXPECT_SOFT_EQ(1e2, calc(Energy{1e2}));
        EXPECT_SOFT_EQ(1e5 - 1e-6, calc(Energy{1e5 - 1e-6}));
        EXPECT_SOFT_EQ(1e5, calc(Energy{1e5}));

        // Test access by index
        EXPECT_SOFT_EQ(1.0, calc[0]);
        EXPECT_SOFT_EQ(1e2, calc[2]);
        EXPECT_SOFT_EQ(1e5, calc[5]);

        // Test between grid points
        EXPECT_SOFT_EQ(5, calc(Energy{5}));
        EXPECT_SOFT_EQ(5e2, calc(Energy{5e2}));
        EXPECT_SOFT_EQ(5e4, calc(Energy{5e4}));

        // Test out-of-bounds
        EXPECT_SOFT_EQ(1.0, calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(1e5, calc(Energy{1e7}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1.0, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e5, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineCalculatorTest, quadratic)
{
    auto xs = [](real_type energy) { return real_type{0.1} * ipow<2>(energy); };

    inp::UniformGrid grid;
    grid.x = {1e-3, 1e2};
    grid.y = {xs(1e-3), xs(1e-2), xs(1e-1), xs(1), xs(1e1), xs(1e2)};

    for (size_type order = 2; order < 5; ++order)
    {
        SCOPED_TRACE("order=" + std::to_string(order));

        grid.interpolation.order = order;
        this->build(grid);

        SplineCalculator calc(this->data().lower, this->values());

        for (real_type e : {1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 1e1, 5e1, 1e2})
        {
            // Interpolation in the construction means small failures in
            // single-precision mode
            EXPECT_SOFT_NEAR(xs(e), calc(Energy{e}), coarse_eps);
        }

        // Test access by index
        EXPECT_SOFT_EQ(xs(1e-3), calc[0]);
        EXPECT_SOFT_EQ(xs(1e-1), calc[2]);
        EXPECT_SOFT_EQ(xs(1e2), calc[5]);

        // Test out-of-bounds
        EXPECT_SOFT_EQ(xs(1e-3), calc(Energy{1e-5}));
        EXPECT_SOFT_EQ(xs(1e2), calc(Energy{1e5}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e2, value_as<Energy>(calc.energy_max()));
    }
}

TEST_F(SplineCalculatorTest, cubic)
{
    auto xs
        = [](real_type energy) { return real_type{0.01} * ipow<3>(energy); };

    inp::UniformGrid grid;
    grid.x = {1e-3, 1e4};

    auto loge_grid = UniformGridData::from_bounds(
        {std::log(grid.x[Bound::lo]), std::log(grid.x[Bound::hi])}, 8);
    UniformGrid loge(loge_grid);
    for (auto i : range(loge.size()))
    {
        grid.y.push_back(xs(std::exp(loge[i])));
    }

    for (size_type order = 3; order < 5; ++order)
    {
        grid.interpolation.order = order;
        this->build(grid);

        SplineCalculator calc(this->data().lower, this->values());

        for (real_type e : {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 1e2, 5e2, 1e3})
        {
            EXPECT_SOFT_EQ(xs(e), calc(Energy{e})) << "e=" << repr(e);
        }

        // Test access by index
        EXPECT_SOFT_EQ(xs(1e-2), calc[1]);
        EXPECT_SOFT_EQ(xs(1.0), calc[3]);
        EXPECT_SOFT_EQ(xs(1e4), calc[7]);

        // Test out-of-bounds
        EXPECT_SOFT_EQ(xs(1e-3), calc(Energy{0.0001}));
        EXPECT_SOFT_EQ(xs(1e4), calc(Energy{1e7}));

        // Test energy grid bounds
        EXPECT_SOFT_EQ(1e-3, value_as<Energy>(calc.energy_min()));
        EXPECT_SOFT_EQ(1e4, value_as<Energy>(calc.energy_max()));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
