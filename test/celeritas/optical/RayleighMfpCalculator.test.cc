//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RayleighMfpCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/RayleighMfpCalculator.hh"

#include "celeritas/mat/MaterialParams.hh"

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
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;

    void SetUp() override {}

    SPConstCoreMaterials build_core_materials() const
    {
        ::celeritas::MaterialParams::Input input;

        static real_type const material_temperatures[]
            = {283.15, 300.0, 283.15, 200, 300.0};

        // Unused element - only to pass checks
        input.elements.push_back(::celeritas::MaterialParams::ElementInput{
            AtomicNumber{1}, units::AmuMass{1}, {}, "fake"});

        for (auto i : range(size_type{5}))
        {
            // Only temperature is relevant information
            input.materials.push_back(
                ::celeritas::MaterialParams::MaterialInput{
                    0,
                    material_temperatures[i] * units::kelvin,
                    MatterState::solid,
                    {},
                    std::to_string(i).c_str()});

            // mock MaterialId == OpticalMaterialId
            input.mat_to_optical.push_back(OpticalMaterialId{i});
        }

        return std::make_shared<::celeritas::MaterialParams const>(
            std::move(input));
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Check calculated MFPs match expected ones
TEST_F(RayleighMfpCalculatorTest, mfp_table)
{
    static std::vector<std::vector<real_type>> expected_tables
        = {{1189584.7068151, 682569.13017288, 343507.60086802},
           {12005096.767467, 6888377.4406869, 3466623.2384762},
           {1189584.7068151, 277.60444893823},
           {11510.805603078, 322.70360179716, 4.230373664558},
           {12005096.767467, 2801.539271218}};

    auto core_materials = this->build_core_materials();

    for (auto opt_mat : range(OpticalMaterialId(import_materials().size())))
    {
        auto const& rayleigh = import_materials()[opt_mat.get()].rayleigh;

        RayleighMfpCalculator calc_mfp(
            MaterialView(this->optical_materials()->host_ref(), opt_mat),
            rayleigh,
            ::celeritas::MaterialView(core_materials->host_ref(),
                                      ::celeritas::MaterialId(opt_mat.get())));

        auto energies = calc_mfp.grid().values();
        auto const& table = expected_tables[opt_mat.get()];

        ASSERT_EQ(energies.size(), table.size());

        std::vector<real_type> expected_mfps(energies.size(), 0);
        std::vector<real_type> mfps(energies.size(), 0);
        for (auto i : range(energies.size()))
        {
            expected_mfps[i] = table[i] * units::Centimeter::value();
            mfps[i] = calc_mfp(units::MevEnergy{energies[i]});
        }

        EXPECT_VEC_SOFT_EQ(expected_mfps, mfps);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
