//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/WentzelVIMsc.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/em/params/WentzelOKVIParams.hh"
#include "celeritas/em/params/WentzelVIMscParams.hh"
#include "celeritas/em/xs/WentzelMacroXsCalculator.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "MscTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
namespace
{
real_type to_inv_cm(real_type xs_native)
{
    return native_value_to<units::InvCmXs>(xs_native).value();
}

}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class WentzelVIMscTest : public MscTestBase
{
    void SetUp() override
    {
        // Load Wentzel VI MSC data
        msc_params_ = WentzelVIMscParams::from_import(*this->particle(),
                                                      this->imported_data());
        ASSERT_TRUE(msc_params_);

        // Default to using both single and multiple Coulomb scattering
        WentzelOKVIParams::Options options;
        options.is_combined = true;
        options.polar_angle_limit = 0.15;
        wentzel_params_ = std::make_shared<WentzelOKVIParams>(
            this->material(), this->particle(), options);
        ASSERT_TRUE(wentzel_params_);

        mat_id_ = this->material()->find_material("G4_STAINLESS-STEEL");
        CELER_ASSERT(mat_id_);
    }

  protected:
    std::shared_ptr<WentzelVIMscParams const> msc_params_;
    std::shared_ptr<WentzelOKVIParams const> wentzel_params_;
    PhysMatId mat_id_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(WentzelVIMscTest, params)
{
    auto const& data = msc_params_->host_ref();
    EXPECT_SOFT_EQ(1e2, value_as<MevEnergy>(data.params.low_energy_limit));
    EXPECT_SOFT_EQ(1e8, value_as<MevEnergy>(data.params.high_energy_limit));
    EXPECT_EQ(1.25, data.params.single_scattering_factor);
    EXPECT_EQ(4, data.xs.size());

    auto const& wentzel = wentzel_params_->host_ref();
    EXPECT_TRUE(wentzel.params.is_combined);
    EXPECT_SOFT_EQ(0.98877107793604224, wentzel.params.costheta_limit);
    EXPECT_SOFT_EQ(1, wentzel.params.screening_factor);
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_SOFT_EQ(19468.968608592968, wentzel.params.a_sq_factor);
    }
    EXPECT_EQ(2, wentzel.inv_mass_cbrt_sq.size());
    EXPECT_SOFT_EQ(9.947409502757395e-1,
                   wentzel.inv_mass_cbrt_sq[PhysMatId(0)]);
    EXPECT_SOFT_EQ(6.8867357655995998e-2,
                   wentzel.inv_mass_cbrt_sq[PhysMatId(1)]);
}

TEST_F(WentzelVIMscTest, TEST_IF_CELERITAS_DOUBLE(total_xs))
{
    auto const& data = msc_params_->host_ref();
    auto const& wentzel = wentzel_params_->host_ref();

    MaterialView material = this->material()->get(mat_id_);

    std::vector<real_type> xs;
    std::vector<real_type> costheta_limit;

    for (real_type energy : {1e2, 1e3, 1e4, 1e6, 1e8})
    {
        ParticleTrackView particle
            = this->make_par_view(pdg::electron(), MevEnergy{energy});
        MevEnergy cutoff
            = this->cutoff()->get(mat_id_).energy(particle.particle_id());

        costheta_limit.push_back(
            1
            - wentzel.params.a_sq_factor * wentzel.inv_mass_cbrt_sq[mat_id_]
                  / value_as<units::MevMomentumSq>(particle.momentum_sq()));

        // The cross section is zero if cos(theta) is above the maximum of the
        // cosine of the polar angle limit (i.e. if single Coulomb scattering
        // is used) and \c 1 - a_sq_factor * inv_mass_cbrt_sq / momentum_sq
        for (real_type theta : {0.0, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.15})
        {
            WentzelMacroXsCalculator calc_xs(
                particle, material, data, wentzel, cutoff);
            xs.push_back(to_inv_cm(calc_xs(std::cos(theta))));
        }
    }

    static double const expected_costheta_limit[] = {
        0.86727876568524,
        0.99866059244724,
        0.99998659360589,
        0.99999999865922,
        0.99999999999987,
    };
    static double const expected_xs[] = {
        91006.493712975,
        90538.361023805,
        59996.704866643,
        1728.032830082,
        17.052848013023,
        0.094248036612716,
        0,
        90817.17035679,
        60099.84698232,
        1743.5607359136,
        17.277463624995,
        0.1648116866444,
        0,
        0,
        90797.843158111,
        1745.1255206409,
        17.293222062587,
        0.16496768570159,
        0,
        0,
        0,
        90795.712682346,
        0.16498499439764,
        0,
        0,
        0,
        0,
        0,
        90795.888575989,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    EXPECT_VEC_SOFT_EQ(expected_costheta_limit, costheta_limit);
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
