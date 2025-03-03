//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/ValueGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/ValueGridBuilder.hh"

#include <memory>
#include <vector>

#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/grid/XsGridInserter.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using std::make_shared;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ValueGridBuilderTest : public Test
{
  public:
    using SPConstBuilder = std::shared_ptr<ValueGridBuilder const>;
    using VecBuilder = std::vector<SPConstBuilder>;
    using VecDbl = std::vector<double>;
    using Energy = XsCalculator::Energy;
    using GridId = XsGridInserter::GridId;

  protected:
    void build(VecBuilder const& entries)
    {
        CELER_EXPECT(!entries.empty());

        // Insert
        XsGridInserter insert(&real_storage, &grid_storage);
        for (SPConstBuilder const& b : entries)
        {
            b->build(insert);
        }
        real_ref = real_storage;
    }

    Collection<real_type, Ownership::value, MemSpace::host> real_storage;
    Collection<real_type, Ownership::const_reference, MemSpace::host> real_ref;
    Collection<XsGridRecord, Ownership::value, MemSpace::host> grid_storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridBuilderTest, xs_grid)
{
    using Builder_t = ValueGridXsBuilder;
    using GridInput = ValueGridXsBuilder::GridInput;

    VecBuilder entries;
    {
        entries.push_back(make_shared<Builder_t>(
            GridInput{1e1, 1e2, {0.1, 0.2 * 1e2}},
            GridInput{1e2, 1e3, {0.2 * 1e2, 0.3 * 1e3}}));
    }
    {
        double const lambda_energy[] = {1e-3, 1e-2, 1e-1};
        double const lambda[] = {10, 1, .1};
        double const lambda_prim_energy[] = {1e-1, 1e0, 10};
        double const lambda_prim[] = {.1 * 1e-1, .01 * 1, .001 * 10};

        entries.push_back(Builder_t::from_geant(
            lambda_energy, lambda, lambda_prim_energy, lambda_prim));
    }
    {
        entries.push_back(make_shared<Builder_t>(
            GridInput{1e-4, 1, VecDbl(18)}, GridInput{1, 1e8, VecDbl(38)}));
    }

    // Build
    this->build(entries);

    // Test results using the physics calculator
    ASSERT_EQ(3, grid_storage.size());
    {
        XsCalculator calc_xs(grid_storage[GridId{0}], real_ref);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
    {
        XsCalculator calc_xs(grid_storage[GridId{1}], real_ref);
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
        entries.push_back(make_shared<Builder_t>(1e1, 1e3, VecDbl{.1, .2, .3}));
    }

    // Build
    this->build(entries);

    // Test results using the physics calculator
    ASSERT_EQ(1, grid_storage.size());
    {
        XsCalculator calc_xs(grid_storage[GridId{0}], real_ref);
        EXPECT_SOFT_EQ(0.1, calc_xs(Energy{1e1}));
        EXPECT_SOFT_EQ(0.2, calc_xs(Energy{1e2}));
        EXPECT_SOFT_EQ(0.3, calc_xs(Energy{1e3}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
