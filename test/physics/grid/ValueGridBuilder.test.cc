//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/ValueGridBuilder.hh"

#include <memory>
#include <vector>
#include "physics/grid/XsCalculator.hh"
#include "physics/grid/ValueGridInserter.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using std::make_shared;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ValueGridBuilderTest : public celeritas::Test
{
  public:
    using SPConstBuilder = std::shared_ptr<const ValueGridBuilder>;
    using VecBuilder     = std::vector<SPConstBuilder>;
    using VecReal        = std::vector<real_type>;
    using Energy         = XsCalculator::Energy;
    using XsIndex        = ValueGridInserter::XsIndex;

  protected:
    void build(const VecBuilder& entries)
    {
        CELER_EXPECT(!entries.empty());

        // Insert
        ValueGridInserter insert(&real_storage, &grid_storage);
        for (const SPConstBuilder& b : entries)
        {
            b->build(insert);
        }
        real_ref = real_storage;
    }

    Collection<real_type, Ownership::value, MemSpace::host> real_storage;
    Collection<real_type, Ownership::const_reference, MemSpace::host> real_ref;
    Collection<XsGridData, Ownership::value, MemSpace::host> grid_storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridBuilderTest, xs_grid)
{
    using Builder_t = ValueGridXsBuilder;

    VecBuilder entries;
    {
        entries.push_back(make_shared<Builder_t>(
            1e1, 1e2, 1e3, VecReal{.1, .2 * 1e2, .3 * 1e3}));
    }
    {
        const real_type lambda_energy[]      = {1e-3, 1e-2, 1e-1};
        const real_type lambda[]             = {10, 1, .1};
        const real_type lambda_prim_energy[] = {1e-1, 1e0, 10};
        const real_type lambda_prim[]        = {.1 * 1e-1, .01 * 1, .001 * 10};

        entries.push_back(Builder_t::from_geant(
            lambda_energy, lambda, lambda_prim_energy, lambda_prim));
    }
    {
        entries.push_back(make_shared<Builder_t>(1e-4, 1, 1e8, VecReal(55)));
    }

    // Build
    this->build(entries);

    // Test results using the physics calculator
    ASSERT_EQ(3, grid_storage.size());
    {
        XsCalculator calc_xs(grid_storage[XsIndex{0}], real_ref);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
    {
        XsCalculator calc_xs(grid_storage[XsIndex{1}], real_ref);
        EXPECT_SOFT_EQ(10., calc_xs(Energy{1e-3}));
        EXPECT_SOFT_EQ(1., calc_xs(Energy{1e-2}));
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e-1}));
        EXPECT_SOFT_EQ(0.01, calc_xs(Energy{1e0}));
        EXPECT_SOFT_EQ(0.001, calc_xs(Energy{1e1}));
    }
}

TEST_F(ValueGridBuilderTest, log_grid)
{
    using Builder_t = ValueGridLogBuilder;

    VecBuilder entries;
    {
        entries.push_back(
            make_shared<Builder_t>(1e1, 1e3, VecReal{.1, .2, .3}));
    }

    // Build
    this->build(entries);

    // Test results using the physics calculator
    ASSERT_EQ(1, grid_storage.size());
    {
        XsCalculator calc_xs(grid_storage[XsIndex{0}], real_ref);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
}

TEST_F(ValueGridBuilderTest, DISABLED_generic_grid)
{
    using Builder_t = ValueGridGenericBuilder;

    VecBuilder entries;
    {
        entries.push_back(
            make_shared<Builder_t>(VecReal{.1, .2, .3}, VecReal{1, 2, 3}));
        entries.push_back(make_shared<Builder_t>(VecReal{1e-2, 1e-1, 1},
                                                 VecReal{1, 2, 3},
                                                 Interp::log,
                                                 Interp::linear));
    }

    // Build
    this->build(entries);

    // Test results using the physics calculator
    ASSERT_EQ(2, grid_storage.size());
    {
        XsCalculator calc_xs(grid_storage[XsIndex{0}], real_ref);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
}
