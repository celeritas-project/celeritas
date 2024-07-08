//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalModelBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/OpticalModelBuilder.hh"

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"
#include "celeritas/optical/model/OpticalRayleighModel.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class OpticalModelBuilderTest : public ::celeritas::test::Test
{
  protected:
    using SPModel = std::shared_ptr<OpticalModel>;

    void SetUp() override
    {
        ScopedRootErrorHandler scoped_root_error_;
        RootImporter import_from_root(
            Test::test_data_path("celeritas", "four-steel-slabs.root").c_str());
        import_data_ = import_from_root();
    }

    ImportData import_data_;
};

TEST_F(OpticalModelBuilderTest, build_absorption)
{
    OpticalModelBuilderOptions options{};
    OpticalModelBuilder model_builder{import_data_, options};

    ActionId start_id{0};
    auto iter = OpticalModelBuilder::ActionIdIter{start_id};

    SPModel model = model_builder(ImportOpticalModelClass::absorption, iter);
    CELER_ASSERT(model);

    CELER_EXPECT(*iter == ActionId{1});
    CELER_EXPECT(start_id == model->action_id());
    CELER_EXPECT(std::dynamic_pointer_cast<AbsorptionModel>(model));
}

TEST_F(OpticalModelBuilderTest, build_rayleigh)
{
    OpticalModelBuilderOptions options{};
    OpticalModelBuilder model_builder{import_data_, options};

    ActionId start_id{0};
    auto iter = OpticalModelBuilder::ActionIdIter{start_id};

    SPModel model = model_builder(ImportOpticalModelClass::rayleigh, iter);
    CELER_ASSERT(model);

    CELER_EXPECT(*iter == ActionId{1});
    CELER_EXPECT(start_id == model->action_id());
    CELER_EXPECT(std::dynamic_pointer_cast<OpticalRayleighModel>(model));
}

TEST_F(OpticalModelBuilderTest, build_wls) {}

TEST_F(OpticalModelBuilderTest, build_all)
{
    OpticalModelBuilderOptions options{};
    OpticalModelBuilder model_builder{import_data_, options};

    ActionId start_id{0};
    auto iter = OpticalModelBuilder::ActionIdIter{start_id};

    std::vector<SPModel> models;

    for (auto iomc : OpticalModelBuilder::get_all_model_classes())
    {
        models.emplace_back(model_builder(iomc, start_id));
    }

    CELER_ASSERT(models.size() == iter->get());

    for (auto id : range(ActionId{models.size()}))
    {
        auto const& model = models[id.get()];
        CELER_ASSERT(model);
        CELER_EXPECT(model->action_id() == id);
    }
}

TEST_F(OpticalModelBuilderTest, warn_and_ignore)
{
    OpticalModelBuilder::UserBuildMap user_map{
        {ImportOpticalModelClass::absorption,
         WarnAndIgnoreOpticalModel{ImportOpticalModelClass::absorption}}};

    OpticalModelBuilderOptions options{};
    OpticalModelBuilder model_builder{import_data_, user_map, options};

    ActionId start_id{0};
    auto iter = OpticalModelBuilder::ActionIdIter{start_id};

    SPModel model = model_builder(ImportOpticalModelClass::absorption, iter);

    // Shouldn't get a model when it's been ignored
    CELER_EXPECT(!model);

    // Action ID also shouldn't change
    CELER_EXPECT(*iter == ActionId{0});
    CELER_EXPECT(*iter == start_id);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
