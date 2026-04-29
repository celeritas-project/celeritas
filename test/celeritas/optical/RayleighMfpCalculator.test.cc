//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/RayleighMfpCalculator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/model/RayleighMfpCalculator.hh"

#include "celeritas/mat/MaterialParams.hh"

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

struct MeterCubedPerMeV
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return ipow<3>(units::meter) / units::Mev::value();
    }

    static char const* label() { return "m^3/MeV"; }
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RayleighMfpCalculatorTest : public OpticalMockTestBase
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Check calculated MFPs match expected ones
TEST_F(RayleighMfpCalculatorTest, mfp_table)
{
    using Compressibility = RealQuantity<MeterCubedPerMeV>;

    static constexpr auto expected_mfps = native_array_from<units::CmLength>(
        // clang-format off
        1189584.7068151, 682569.13017288, 343507.60086802, 12005096.767467,
        6888377.4406869, 3466623.2384762, 1189584.7068151, 277.60444893823,
        11510.805603078, 322.70360179716, 4.230373664558, 12005096.767467,
        2801.539271218
        // clang-format on
    );

    auto core_materials = this->material();

    size_type num_materials = 5;
    CELER_ASSERT(this->optical_material()->num_materials() == num_materials);
    std::vector<inp::OpticalRayleighAnalytic> rayleigh(num_materials);
    rayleigh[0].compressibility = native_value_from(Compressibility{7.658e-23});
    rayleigh[1].scale_factor = 1.7;
    rayleigh[1].compressibility = native_value_from(Compressibility{4.213e-24});
    rayleigh[2].compressibility = native_value_from(Compressibility{7.658e-23});
    rayleigh[3].scale_factor = 2;
    rayleigh[3].compressibility = native_value_from(Compressibility{1e-20});
    rayleigh[4].scale_factor = 1.7;
    rayleigh[4].compressibility = native_value_from(Compressibility{4.213e-24});

    std::vector<real_type> mfps;
    mfps.reserve(expected_mfps.size());

    for (auto opt_mat : range(OptMatId(num_materials)))
    {
        RayleighMfpCalculator calc_mfp(
            this->optical_material()->get(opt_mat),
            rayleigh[opt_mat.get()],
            this->material()->get(::celeritas::PhysMatId(opt_mat.get())));

        auto energies = calc_mfp.grid().values();
        for (auto i : range(energies.size()))
        {
            mfps.push_back(calc_mfp(units::MevEnergy{energies[i]}));
        }
    }

    EXPECT_VEC_SOFT_EQ(expected_mfps, mfps);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
