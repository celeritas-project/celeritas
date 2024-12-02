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
        EXPECT_EQ(materials.size(), model.mfp_table.size());

        for (auto const& mfp : model.mfp_table)
        {
            EXPECT_EQ(ImportPhysicsVectorType::free, mfp.vector_type);
            EXPECT_TRUE(mfp);
        }
    }

    // Check materials

    for (auto const& material : materials)
    {
        EXPECT_TRUE(material.properties);
        EXPECT_TRUE(material.rayleigh);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
