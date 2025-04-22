//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedMaterials.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ImportedMaterials.hh"

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

class ImportedMaterialsTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override
    {
        imported_materials
            = ImportedMaterials::from_import(this->imported_data());
    }

    std::shared_ptr<ImportedMaterials const> imported_materials;
};

//---------------------------------------------------------------------------//
// Check mock data is correctly mapped after from import
TEST_F(ImportedMaterialsTest, simple)
{
    ASSERT_TRUE(imported_materials);

    for (auto opt_mat : range(OptMatId{imported_materials->num_materials()}))
    {
        // Rayleigh data
        {
            auto const& expected_rayleigh
                = this->imported_data().optical_materials[opt_mat.get()].rayleigh;
            auto const& imported_rayleigh
                = this->imported_materials->rayleigh(opt_mat);

            EXPECT_EQ(expected_rayleigh.scale_factor,
                      imported_rayleigh.scale_factor);
            EXPECT_EQ(expected_rayleigh.compressibility,
                      imported_rayleigh.compressibility);
        }

        // Wavelength shifting
        {
            auto const& expected_wls
                = this->imported_data().optical_materials[opt_mat.get()].wls;
            auto const& imported_wls = this->imported_materials->wls(opt_mat);

            EXPECT_EQ(expected_wls.mean_num_photons,
                      imported_wls.mean_num_photons);
            EXPECT_EQ(expected_wls.time_constant, imported_wls.time_constant);
            EXPECT_GRID_EQ(expected_wls.component, imported_wls.component);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
