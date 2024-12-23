//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MfpBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/MfpBuilder.hh"

#include "OpticalMockTestBase.hh"
#include "ValidationUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MfpBuilderTest : public OpticalMockTestBase
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Check MFP tables are built with correct structure from imported data
TEST_F(MfpBuilderTest, construct_tables)
{
    OwningGridAccessor storage;

    std::vector<ItemRange<OwningGridAccessor::Grid>> tables;
    auto const& models = this->imported_data().optical_models;

    // Build MFP tables from imported data
    for (auto const& model : models)
    {
        auto build = storage.create_mfp_builder();

        for (auto const& mfp : model.mfp_table)
        {
            build(mfp);
        }

        tables.push_back(build.grid_ids());
    }

    ASSERT_EQ(models.size(), tables.size());

    // Check each MFP table has been built correctly
    for (auto table_id : range(tables.size()))
    {
        EXPECT_TABLE_EQ(models[table_id].mfp_table, storage(tables[table_id]));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
