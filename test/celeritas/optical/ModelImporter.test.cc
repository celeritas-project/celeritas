//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ModelImporter.hh"

#include <algorithm>

#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/ModelBuilder.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/RayleighModel.hh"

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

class ModelImporterTest : public MockImportedData
{
  protected:
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterial
        = std::shared_ptr<::celeritas::MaterialParams const>;

    static void SetUpTestCase()
    {
        auto& data = import_data();
        data.optical_models = import_models();
        data.optical_materials = import_materials();
        data.units = units::NativeTraits::label();
    }

    ModelImporter
    build_importer(ModelImporter::UserBuildMap const& user_build = {})
    {
        return ModelImporter(import_data(),
                             this->optical_materials(),
                             this->core_materials(),
                             user_build);
    }

    static ImportData& import_data()
    {
        static ImportData import_data_;
        return import_data_;
    }

    template<class ModelT>
    void check_import_model(ModelImporter const& model_importer,
                            ImportModelClass imc)
    {
        // Create a model builder
        auto build = model_importer(imc);
        ASSERT_TRUE(build);

        // Create a model
        ActionId action_id{0};
        auto model = (*build)(action_id);
        ASSERT_TRUE(model);

        // Check we have the correct model
        EXPECT_EQ(action_id, model->action_id());
        EXPECT_TRUE(std::dynamic_pointer_cast<ModelT const>(model));

        // Get expected MFP tables
        auto expected_iter = std::find_if(
            this->import_models().begin(),
            this->import_models().end(),
            [imc](auto const& m) { return m.model_class == imc; });
        ASSERT_FALSE(expected_iter == this->import_models().end());

        // Build imported tables
        auto mfp_builder = this->create_mfp_builder();
        for (auto mat : range(OpticalMaterialId{
                 this->optical_materials()->num_materials()}))
        {
            model->build_mfps(mat, mfp_builder);
        }

        // Check tables (i.e. models have correct data after being built)
        this->check_built_table_exact(expected_iter->mfp_table,
                                      mfp_builder.grid_ids());
    }
};

//---------------------------------------------------------------------------//
// Test building absorption
TEST_F(ModelImporterTest, build_absorption)
{
    auto model_importer = this->build_importer();

    this->check_import_model<AbsorptionModel>(model_importer,
                                              ImportModelClass::absorption);
}

//---------------------------------------------------------------------------//
// Test building Rayleigh scattering
TEST_F(ModelImporterTest, build_rayleigh)
{
    auto model_importer = this->build_importer();

    this->check_import_model<RayleighModel>(model_importer,
                                            ImportModelClass::rayleigh);
}

//---------------------------------------------------------------------------//
// Test building WLS
TEST_F(ModelImporterTest, build_wls) {}

//---------------------------------------------------------------------------//
// Test user ignore options
TEST_F(ModelImporterTest, warn_and_ignore)
{
    ModelImporter::UserBuildMap user_map{
        {ImportModelClass::absorption,
         WarnAndIgnoreModel{ImportModelClass::absorption}},
        {ImportModelClass::wls, WarnAndIgnoreModel{ImportModelClass::wls}},
    };

    auto model_importer = this->build_importer(user_map);

    EXPECT_FALSE(model_importer(ImportModelClass::absorption));
    this->check_import_model<RayleighModel>(model_importer,
                                            ImportModelClass::rayleigh);
    EXPECT_FALSE(model_importer(ImportModelClass::wls));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
