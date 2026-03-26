//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Absorption.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/interactor/AbsorptionInteractor.hh"
#include "celeritas/optical/model/AbsorptionModel.hh"

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

class AbsorptionInteractorTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

class AbsorptionModelTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override
    {
        model = std::make_shared<AbsorptionModel const>(
            ActionId{0}, this->imported_data().optical_physics.bulk.absorption);
    }

    std::shared_ptr<AbsorptionModel const> model;
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
    EXPECT_EQ(ActionId{0}, model->action_id());
    EXPECT_EQ("absorption", model->label());
    EXPECT_EQ("interact by optical absorption", model->description());
}

//---------------------------------------------------------------------------//
// Check absorption model MFP tables match imported ones
TEST_F(AbsorptionModelTest, interaction_mfp)
{
    OwningGridAccessor storage;

    auto builder = storage.create_mfp_builder();
    for (auto mat : range(OptMatId(this->num_optical_materials())))
    {
        model->build_mfps(mat, builder);
    }

    EXPECT_TABLE_EQ(this->get_mfp_table(ImportModelClass::absorption),
                    storage(builder.grid_ids()));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
