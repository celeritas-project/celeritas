//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ModelImporter.hh"

#include <algorithm>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/ModelBuilder.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/RayleighModel.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class ModelImporterTest : public ::celeritas::test::Test
{
  protected:
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterial
        = std::shared_ptr<::celeritas::MaterialParams const>;

    static void SetUpTestCase()
    {
        ScopedRootErrorHandler scoped_root_error;
        RootImporter import_from_root(
            Test::test_data_path("celeritas", "lar-sphere.root").c_str());
        import_data() = import_from_root();
        core_material()
            = ::celeritas::MaterialParams::from_import(import_data());

        {
            MaterialParams::Input inp;
            inp.properties.reserve(import_data().optical_materials.size());
            for (auto const& mat : import_data().optical_materials)
            {
                inp.properties.push_back(mat.properties);
            }
            inp.volume_to_mat = {OpticalMaterialId{}};

            material() = std::make_shared<MaterialParams const>(std::move(inp));
        }

        CELER_ENSURE(material() && core_material());
    }

    ModelImporter
    build_importer(ModelImporter::UserBuildMap const& user_build = {})
    {
        return ModelImporter(
            import_data(), material(), core_material(), user_build);
    }

    static ImportData& import_data()
    {
        static ImportData import_data_;
        return import_data_;
    }

    static SPConstMaterial& material()
    {
        static SPConstMaterial m;
        return m;
    }

    static SPConstCoreMaterial& core_material()
    {
        static SPConstCoreMaterial m;
        return m;
    }
};

TEST_F(ModelImporterTest, build_absorption)
{
    auto model_importer = this->build_importer();

    auto build = model_importer(ImportModelClass::absorption);
    ASSERT_TRUE(build);

    ActionId action_id{0};
    auto model = (*build)(action_id);
    ASSERT_TRUE(model);

    ASSERT_EQ(action_id, model->action_id());
    ASSERT_TRUE(std::dynamic_pointer_cast<AbsorptionModel const>(model));
}

TEST_F(ModelImporterTest, build_rayleigh)
{
    auto model_importer = this->build_importer();

    auto build = model_importer(ImportModelClass::rayleigh);
    ASSERT_TRUE(build);

    ActionId action_id{0};
    auto model = (*build)(action_id);
    ASSERT_TRUE(model);

    ASSERT_EQ(action_id, model->action_id());
    ASSERT_TRUE(std::dynamic_pointer_cast<RayleighModel const>(model));
}

TEST_F(ModelImporterTest, build_wls) {}

TEST_F(ModelImporterTest, warn_and_ignore)
{
    ModelImporter::UserBuildMap user_map{
        {ImportModelClass::absorption,
         WarnAndIgnoreModel{ImportModelClass::absorption}},
        {ImportModelClass::wls, WarnAndIgnoreModel{ImportModelClass::wls}},
    };

    auto model_importer = this->build_importer(user_map);

    EXPECT_FALSE(model_importer(ImportModelClass::absorption));
    EXPECT_TRUE(model_importer(ImportModelClass::rayleigh));
    EXPECT_FALSE(model_importer(ImportModelClass::wls));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
