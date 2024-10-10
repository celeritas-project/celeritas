//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MockValidation.test.cc
//---------------------------------------------------------------------------//
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

class MockValidationTest : public MockImportedData
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Validate that the mock optical data makes sense
TEST_F(MockValidationTest, validate)
{
    auto const& models = import_models();
    auto const& materials = import_materials();

    EXPECT_EQ(3, models.size());
    EXPECT_EQ(5, materials.size());

    // Check models

    for (auto const& model : models)
    {
        EXPECT_NE(ImportModelClass::size_, model.model_class);
        EXPECT_EQ(materials.size(), model.mfps.size());

        for (auto const& mfp : model.mfps)
        {
            EXPECT_EQ(ImportPhysicsVectorType::free, mfp.vector_type);
            EXPECT_TRUE(mfp);
        }
    }

    // Check IDs correspond to correct imported model

    ASSERT_LT(absorption_id().get(), models.size());
    EXPECT_EQ(ImportModelClass::absorption,
              models[absorption_id().get()].model_class);

    ASSERT_LT(rayleigh_id().get(), models.size());
    EXPECT_EQ(ImportModelClass::rayleigh,
              models[rayleigh_id().get()].model_class);

    ASSERT_LT(wls_id().get(), models.size());
    EXPECT_EQ(ImportModelClass::wls, models[wls_id().get()].model_class);

    // TODO: Check materials
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
