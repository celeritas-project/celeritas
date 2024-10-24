//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MfpBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/MfpBuilder.hh"

#include "MockImportedData.hh"
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

class MfpBuilderTest : public MockImportedData
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Check MFP tables are built with correct structure from imported data
TEST_F(MfpBuilderTest, construct_tables)
{
    std::vector<ItemRange<Grid>> tables;
    auto const& models = this->import_models();

    // Build MFP tables from imported data
    for (auto const& model : models)
    {
        auto build = this->create_mfp_builder();

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
        this->check_built_table_exact(models[table_id].mfp_table, tables[table_id]);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
