//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/ValueGridBuilder.hh"

#include <memory>
#include <vector>
#include "physics/grid/PhysicsGridCalculator.hh"
#include "physics/grid/ValueGridStore.hh"
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
    using Energy         = PhysicsGridCalculator ::Energy;

  protected:
    ValueGridStore build(const VecBuilder& entries) const
    {
        CELER_EXPECT(!entries.empty());

        // Construct sizes
        size_type num_values = 0;
        for (const SPConstBuilder& b : entries)
        {
            CELER_EXPECT(b);
            auto required_storage = b->storage();
            EXPECT_GT(required_storage.second, 0);
            num_values += required_storage.second;
        }

        // Build store
        ValueGridStore store(entries.size(), num_values);
        for (const SPConstBuilder& b : entries)
        {
            b->build(&store);
        }

        CELER_ENSURE(store.size() == store.capacity());
        return store;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridBuilderTest, xs_grid)
{
    using Builder_t = ValueGridXsBuilder;
    using VecReal   = std::vector<real_type>;

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

        entries.push_back(make_shared<Builder_t>(Builder_t::from_geant(
            lambda_energy, lambda, lambda_prim_energy, lambda_prim)));
    }

    // Build
    ValueGridStore store = this->build(entries);
    auto           ptrs  = store.host_pointers();

    // Test results using the physics calculator
    ASSERT_EQ(2, ptrs.size());
    {
        PhysicsGridCalculator calc_xs(ptrs[0]);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
    {
        PhysicsGridCalculator calc_xs(ptrs[1]);
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
    using VecReal   = std::vector<real_type>;

    VecBuilder entries;

    {
        entries.push_back(
            make_shared<Builder_t>(1e1, 1e3, VecReal{.1, .2, .3}));
    }

    // Build
    ValueGridStore store = this->build(entries);
    auto           ptrs  = store.host_pointers();

    // Test results using the physics calculator
    ASSERT_EQ(1, ptrs.size());
    {
        PhysicsGridCalculator calc_xs(ptrs[0]);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
}
