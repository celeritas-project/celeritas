//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Absorption.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/interactor/AbsorptionInteractor.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"

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

class AbsorptionInteractorTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

class AbsorptionModelTest : public ::celeritas::test::Test,
                            public MockImportedData
{
  protected:
    void SetUp() override {}

    //! Construct absorption model from mock data
    std::shared_ptr<AbsorptionModel const> create_model() const
    {
        return std::make_shared<AbsorptionModel const>(
            ActionId{0},
            ImportedModelAdapter{MockImportedData::absorption_id(),
                                 MockImportedData::create_imported_models()});
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Basic test for just absorption interaction
TEST_F(AbsorptionInteractorTest, basic)
{
    // A simple regression test to make sure the interaction is absorbed

    AbsorptionInteractor interact;
    Interaction result = interact();

    // Do a few checks to make sure there's no state
    for ([[maybe_unused]] int i : range(10))
    {
        EXPECT_EQ(Interaction::Action::absorbed, result.action);
    }
}

//---------------------------------------------------------------------------//
// Check model name and description are properly initialized
TEST_F(AbsorptionModelTest, description)
{
    auto model = create_model();

    EXPECT_EQ(ActionId{0}, model->action_id());
    EXPECT_EQ("absorption", model->label());
    EXPECT_EQ("interact by optical absorption", model->description());
}

//---------------------------------------------------------------------------//
// Check absorption model MFP tables match imported ones
TEST_F(AbsorptionModelTest, interaction_mfp)
{
    auto model = create_model();
    auto builder = this->create_mfp_builder();

    model->build_mfps(builder);

    this->check_built_table(
        this->import_models()[this->absorption_id().get()].mfps,
        builder.grid_ids());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
