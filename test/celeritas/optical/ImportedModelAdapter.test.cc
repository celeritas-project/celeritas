//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ImportedModelAdapter.hh"

#include <array>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"

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

class ImportedModelAdapterTest : public MockImportedData
{
  protected:
    void SetUp() override {}

    void check_model(ImportOpticalModel const& expected_model,
                     ImportOpticalModel const& imported_model) const
    {
        EXPECT_EQ(expected_model.model_class, imported_model.model_class);
        ASSERT_EQ(expected_model.mfp_table.size(),
                  imported_model.mfp_table.size());
        for (auto mat_id : range(imported_model.mfp_table.size()))
        {
            this->check_mfp(expected_model.mfp_table[mat_id],
                            imported_model.mfp_table[mat_id]);
        }
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Create ImportedModels from mock data
TEST_F(ImportedModelAdapterTest, build_mock)
{
    auto const& expected_models = this->import_models();
    auto imported_models = this->create_imported_models();

    ASSERT_EQ(expected_models.size(), imported_models->num_models());
    for (auto model_id : range(ImportedModelId{imported_models->num_models()}))
    {
        this->check_model(expected_models[model_id.get()],
                          imported_models->model(model_id));
    }
}

//---------------------------------------------------------------------------//
// Check built-in map properly created
TEST_F(ImportedModelAdapterTest, builtin_map)
{
    using IMC = ImportModelClass;
    std::array<IMC, 3> expected_builtin_imcs{
        IMC::absorption, IMC::rayleigh, IMC::wls};

    auto imported_models = this->create_imported_models();

    // Check built-in models match expected ones
    EXPECT_EQ(expected_builtin_imcs.size(), static_cast<size_type>(IMC::size_));

    // Check mapping is correct
    for (auto imc : expected_builtin_imcs)
    {
        auto model_id = imported_models->builtin_model_id(imc);
        ASSERT_LT(model_id, imported_models->num_models());
        EXPECT_EQ(imc, imported_models->model(model_id).model_class);
    }
}

//---------------------------------------------------------------------------//
// Check adapters correctly match MFPs
TEST_F(ImportedModelAdapterTest, adapter_mfps)
{
    auto const& expected_models = this->import_models();
    auto imported_models = this->create_imported_models();

    ASSERT_EQ(expected_models.size(), imported_models->num_models());
    for (auto model_id : range(ImportedModelId{imported_models->num_models()}))
    {
        ImportedModelAdapter adapter(model_id, imported_models);
        auto const& expected_model = expected_models[model_id.get()];

        ASSERT_EQ(expected_model.mfp_table.size(), adapter.num_materials());
        for (auto mat_id : range(OpticalMaterialId{adapter.num_materials()}))
        {
            this->check_mfp(expected_model.mfp_table[mat_id.get()],
                            adapter.mfp(mat_id));
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas