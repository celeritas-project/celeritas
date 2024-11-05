//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RayleighMfpCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/RayleighMfpCalculator.hh"

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

class RayleighMfpCalculatorTest : public MockImportedData
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Check calculated MFPs match expected ones
TEST_F(RayleighMfpCalculatorTest, mfp_table)
{
    static std::vector<std::vector<real_type>> expected_energies = {
        {1.098177, 1.256172, 1.484130},
        {1.098177, 1.256172, 1.484130},
        {1.098177, 6.812319},
        {1, 2, 5},
        {1.098177, 6.812319},
    };

    static std::vector<std::vector<real_type>> expected_mfps
        = {{1189584.7068151, 682569.13017288, 343507.60086802},
           {12005096.767467, 6888377.4406869, 3466623.2384762},
           {1189584.7068151, 277.60444893823},
           {11510.805603078, 322.70360179716, 4.230373664558},
           {12005096.767467, 2801.539271218}};

    auto expected_mfp_tables
        = detail::convert_vector_units<detail::ElectronVolt, units::Centimeter>(
            expected_energies, expected_mfps);

    static real_type const material_temperatures[] = {283.15 * units::kelvin,
                                                      300.0 * units::kelvin,
                                                      283.15 * units::kelvin,
                                                      200 * units::kelvin,
                                                      300.0 * units::kelvin};

    for (auto opt_mat : range(OpticalMaterialId(import_materials().size())))
    {
        auto const& rayleigh = import_materials()[opt_mat.get()].rayleigh;

        RayleighMfpCalculator calc_mfp(
            MaterialView(this->optical_materials()->host_ref(), opt_mat),
            OpticalRayleighMaterial{rayleigh.scale_factor,
                                    rayleigh.compressibility,
                                    material_temperatures[opt_mat.get()]});

        auto const& expected_table = expected_mfp_tables[opt_mat.get()];

        std::vector<real_type> mfps;
        mfps.reserve(expected_table.x.size());
        for (real_type energy : expected_table.x)
        {
            mfps.push_back(calc_mfp(celeritas::units::MevEnergy{energy}));
        }

        EXPECT_VEC_SOFT_EQ(expected_table.y, mfps);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
