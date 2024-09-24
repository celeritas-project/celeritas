//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedModelAdapter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ImportedModelAdapter.hh"

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalModel.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class ImportedModelsTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        ScopedRootErrorHandler scoped_root_error_;
        RootImporter import_from_root(
            Test::test_data_path("celeritas", "four-steel-slabs.root").c_str());
        import_data_ = import_from_root();
    }

    ImportData import_data_;

    constexpr static std::array<ImportModelClass, 2> expected_builtin_models
        = {ImportModelClass::absorption, ImportModelClass::rayleigh};

    std::unordered_map<ImportModelClass, ModelId>
    builtin_model_id_map(ImportedModels const& imported) const
    {
        EXPECT_LE(this->expected_builtin_models.size(), imported.num_models());

        std::unordered_map<ImportModelClass, ModelId> class_to_id;
        for (auto imc : this->expected_builtin_models)
        {
            for (auto model_id : range(ModelId{imported.num_models()}))
            {
                EXPECT_TRUE(class_to_id.find(imc) == class_to_id.end());
                class_to_id.insert({imc, model_id});
            }
        }

        EXPECT_EQ(this->expected_builtin_models.size(), class_to_id.size());

        return class_to_id;
    }
};

TEST_F(ImportedModelsTest, from_import)
{
    auto imported_models = ImportedModels::from_import(this->import_data_);
    ASSERT_TRUE(imported_models);

    EXPECT_EQ(this->import_data_.optical_materials.size(),
              imported_models->num_materials());

    auto class_to_id = this->builtin_model_id_map(*imported_models);

    {
        // Absorption
        ASSERT_TRUE(class_to_id.find(ImportModelClass::absorption)
                    != class_to_id.end());
        auto model = imported_models->model(
            class_to_id[ImportModelClass::absorption]);

        EXPECT_EQ(0, model.mfps.size());

        for (auto mat_id :
             range(OpticalMaterialId{imported_models->num_materials()}))
        {
            EXPECT_TRUE(imported_models->material(mat_id).absorption);
        }
    }

    {
        // Rayleigh
        ASSERT_TRUE(class_to_id.find(ImportModelClass::rayleigh)
                    != class_to_id.end());
        auto model
            = imported_models->model(class_to_id[ImportModelClass::rayleigh]);

        EXPECT_EQ(0, model.mfps.size());

        for (auto mat_id :
             range(OpticalMaterialId{imported_models->num_materials()}))
        {
            EXPECT_TRUE(imported_models->material(mat_id).rayleigh);
        }
    }
}

TEST_F(ImportedModelsTest, custom_builtin_models)
{
    {
        // Mock absorption model data
        ImportOpticalModel model;
        model.model_class = ImportModelClass::absorption;
        for (auto mat_id :
             range(OpticalMaterialId{import_data_.optical_materials.size()}))
        {
            model.mfps.push_back(
                ImportPhysicsVector{ImportPhysicsVectorType::linear,
                                    std::vector<double>{1e-3, 1e5},
                                    std::vector<double>{2e-7, 1e2}});
        }
        import_data_.optical_models.push_back(std::move(model));
    }

    auto imported_models = ImportedModels::from_import(import_data_);
    ASSERT_TRUE(imported_models);

    EXPECT_EQ(this->import_data_.optical_materials.size(),
              imported_models->num_materials());

    auto class_to_id = this->builtin_model_id_map(*imported_models);

    {
        // Check mock absorption data is available
        ASSERT_TRUE(class_to_id.find(ImportModelClass::absorption)
                    != class_to_id.end());
        auto model = imported_models->model(
            class_to_id[ImportModelClass::absorption]);

        EXPECT_EQ(imported_models->num_materials(), model.mfps.size());

        auto const& expected_mfps
            = this->import_data_.optical_models.back().mfps;

        for (auto mat :
             range(OpticalMaterialId{imported_models->num_materials()}))
        {
            EXPECT_VEC_EQ(expected_mfps[mat.get()].x, model.mfps[mat.get()].x);
            EXPECT_VEC_EQ(expected_mfps[mat.get()].y, model.mfps[mat.get()].y);
        }
    }
}

TEST_F(ImportedModelsTest, adapter_material_mfp)
{
    auto imported_models = ImportedModels::from_import(import_data_);
    ASSERT_TRUE(imported_models);

    EXPECT_EQ(this->import_data_.optical_materials.size(),
              imported_models->num_materials());

    auto class_to_id = this->builtin_model_id_map(*imported_models);

    ASSERT_TRUE(class_to_id.find(ImportModelClass::absorption)
                != class_to_id.end());
    ImportedModelAdapter absorption(imported_models,
                                    class_to_id[ImportModelClass::absorption]);

    ASSERT_TRUE(class_to_id.find(ImportModelClass::rayleigh)
                != class_to_id.end());
    ImportedModelAdapter rayleigh(imported_models,
                                  class_to_id[ImportModelClass::rayleigh]);

    for (auto mat : range(OpticalMaterialId{imported_models->num_materials()}))
    {
        EXPECT_EQ(&imported_models->material(mat).absorption.absorption_length,
                  absorption.material_mfp(mat));
        EXPECT_EQ(absorption.material_mfp(mat), absorption.preferred_mfp(mat));

        EXPECT_EQ(&imported_models->material(mat).rayleigh.mfp,
                  rayleigh.material_mfp(mat));
        EXPECT_EQ(rayleigh.material_mfp(mat), rayleigh.preferred_mfp(mat));
    }
}

TEST_F(ImportedModelsTest, adapter_model_mfp)
{
    {
        // Mock absorption model data
        ImportOpticalModel model;
        model.model_class = ImportModelClass::absorption;
        for (auto mat_id :
             range(OpticalMaterialId{import_data_.optical_materials.size()}))
        {
            model.mfps.push_back(
                ImportPhysicsVector{ImportPhysicsVectorType::linear,
                                    std::vector<double>{1e-3, 1e5},
                                    std::vector<double>{2e-7, 1e2}});
        }
        import_data_.optical_models.push_back(std::move(model));
    }

    auto imported_models = ImportedModels::from_import(import_data_);
    ASSERT_TRUE(imported_models);

    EXPECT_EQ(this->import_data_.optical_materials.size(),
              imported_models->num_materials());

    auto class_to_id = this->builtin_model_id_map(*imported_models);

    ASSERT_TRUE(class_to_id.find(ImportModelClass::absorption)
                != class_to_id.end());
    ImportedModelAdapter absorption(imported_models,
                                    class_to_id[ImportModelClass::absorption]);
    EXPECT_EQ(imported_models->num_materials(), absorption.num_materials());

    ASSERT_TRUE(class_to_id.find(ImportModelClass::rayleigh)
                != class_to_id.end());
    ImportedModelAdapter rayleigh(imported_models,
                                  class_to_id[ImportModelClass::rayleigh]);
    EXPECT_EQ(imported_models->num_materials(), rayleigh.num_materials());

    for (auto mat : range(OpticalMaterialId{imported_models->num_materials()}))
    {
        EXPECT_EQ(
            &imported_models->model(class_to_id[ImportModelClass::absorption])
                 .mfps[mat.get()],
            absorption.material_mfp(mat));
        EXPECT_EQ(absorption.imported_mfp(mat), absorption.preferred_mfp(mat));

        EXPECT_EQ(&imported_models->material(mat).rayleigh.mfp,
                  rayleigh.material_mfp(mat));
        EXPECT_EQ(rayleigh.material_mfp(mat), rayleigh.preferred_mfp(mat));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
