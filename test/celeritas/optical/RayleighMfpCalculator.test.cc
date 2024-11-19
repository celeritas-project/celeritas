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
    static std::vector<std::vector<real_type>> expected_tables
        = {{1189584.7068151, 682569.13017288, 343507.60086802},
           {12005096.767467, 6888377.4406869, 3466623.2384762},
           {1189584.7068151, 277.60444893823},
           {11510.805603078, 322.70360179716, 4.230373664558},
           {12005096.767467, 2801.539271218}};

    auto core_materials = this->core_materials();

    for (auto opt_mat : range(OpticalMaterialId(import_materials().size())))
    {
        auto const& rayleigh = import_materials()[opt_mat.get()].rayleigh;

        RayleighMfpCalculator calc_mfp(
            this->optical_materials()->get(opt_mat),
            rayleigh,
            this->core_materials()->get(::celeritas::MaterialId(opt_mat.get())));

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
