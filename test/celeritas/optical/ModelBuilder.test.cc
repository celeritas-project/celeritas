//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ModelBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ModelBuilder.hh"

#include <fstream>
#include <iostream>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/Model.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/RayleighModel.hh"

#include "celeritas_test.hh"
#include "../ImportedDataTestBase.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

//---------------------------------------------------------------------------//

class ModelBuilderTest : public celeritas::test::ImportedDataTestBase
{
  protected:
    using SPModel = std::shared_ptr<Model>;

    void SetUp() override
    {
        ScopedRootErrorHandler scoped_root_error_;
        RootImporter import_from_root(
            Test::test_data_path("celeritas", "four-steel-slabs.root").c_str());
        import_data_ = import_from_root();
    }

    ImportData import_data_;

    template<class M>
    void check_one_model(ImportModelClass imc)
    {
        ModelBuilderOptions options{};
        ModelBuilder model_builder{import_data_, options};

        ActionId start_id{0};
        auto iter = ModelBuilder::ActionIdIter{start_id};

        SPModel model = model_builder(imc, iter);
        ASSERT_TRUE(model);

        EXPECT_EQ(*iter, ActionId{1});
        EXPECT_EQ(start_id, model->action_id());
        EXPECT_TRUE(std::dynamic_pointer_cast<M>(model));
    }
};

TEST_F(ModelBuilderTest, build_absorption)
{
    check_one_model<AbsorptionModel>(ImportModelClass::absorption);
}

TEST_F(ModelBuilderTest, build_rayleigh)
{
    check_one_model<RayleighModel>(ImportModelClass::rayleigh);
}

TEST_F(ModelBuilderTest, build_wls)
{
    std::ofstream outfile("absorption.data");

    for (auto opt_mat_id :
         range(OpticalMaterialId{import_data_.optical_materials.size()}))
    {
        outfile << "Optical Material: " << opt_mat_id.get() << "\n";

        auto const& mfp = import_data_.optical_materials[opt_mat_id.get()]
                              .absorption.absorption_length;

        if (mfp)
        {
            outfile << "x:";
            for (double x : mfp.x)
                outfile << " " << x;
            outfile << "\ny:";
            for (double y : mfp.y)
                outfile << " " << y;
            outfile << "\n";
        }
        else
        {
            outfile << "No listed data\n";
        }
        outfile << "\n";
    }
}

TEST_F(ModelBuilderTest, build_all)
{
    ModelBuilderOptions options{};
    ModelBuilder model_builder{import_data_, options};

    ActionId start_id{0};
    auto iter = ModelBuilder::ActionIdIter{start_id};

    std::vector<SPModel> models;

    for (auto imc : ModelBuilder::get_all_model_classes())
    {
        models.emplace_back(model_builder(imc, iter));
    }

    EXPECT_EQ(models.size(), ModelBuilder::get_all_model_classes().size());
    EXPECT_EQ(models.size(), iter->get());

    for (auto id : range(ModelId{models.size()}))
    {
        auto const& model = models[id.get()];
        EXPECT_TRUE(model);
        EXPECT_EQ(model->action_id().get(), id.get());
    }
}

TEST_F(ModelBuilderTest, warn_and_ignore)
{
    ModelBuilder::UserBuildMap user_map{
        {ImportModelClass::absorption,
         WarnAndIgnoreModel{ImportModelClass::absorption}}};

    ModelBuilderOptions options{};
    ModelBuilder model_builder{import_data_, user_map, options};

    ActionId start_id{0};
    auto iter = ModelBuilder::ActionIdIter{start_id};

    SPModel model = model_builder(ImportModelClass::absorption, iter);

    // Shouldn't get a model when it's been ignored
    ASSERT_FALSE(model);

    // Action ID also shouldn't change
    EXPECT_EQ(*iter, ActionId{0});
    EXPECT_EQ(*iter, start_id);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
