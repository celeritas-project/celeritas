//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
        wentzel_params_
            = std::make_shared<WentzelOKVIParams>(this->material(), options);
        ASSERT_TRUE(wentzel_params_);

        mat_id_ = this->material()->find_material("G4_STAINLESS-STEEL");
        CELER_ASSERT(mat_id_);
    }

  protected:
    std::shared_ptr<WentzelVIMscParams const> msc_params_;
    std::shared_ptr<WentzelOKVIParams const> wentzel_params_;
    MaterialId mat_id_;
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
    EXPECT_SOFT_EQ(4.9976257697681963e-8, wentzel.params.a_sq_factor);
}

TEST_F(WentzelVIMscTest, total_xs)
{
    auto const& data = msc_params_->host_ref();
    auto const& wentzel = wentzel_params_->host_ref();

    MaterialView material = this->material()->get(mat_id_);

    std::vector<real_type> xs;

    for (real_type energy : {1e2, 1e3, 1e4, 1e6, 1e8})
    {
        ParticleTrackView particle
            = this->make_par_view(pdg::electron(), MevEnergy{energy});
        MevEnergy cutoff
            = this->cutoff()->get(mat_id_).energy(particle.particle_id());

        // The cross section is zero if theta is above the polar angle limit,
        // i.e. if single Coulomb scattering is used
        for (real_type theta :
             {0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.15, constants::pi / 2})
        {
            WentzelMacroXsCalculator calc_xs(
                particle, material, data, wentzel, cutoff);
            xs.push_back(calc_xs(std::cos(theta)));
        }
    }

    static double const expected_xs[] = {91006.507959232,
                                         59996.714258593,
                                         1728.0331005902,
                                         17.052850682495,
                                         0.60318444563625,
                                         0.094248051366407,
                                         0,
                                         0,
                                         90817.190202029,
                                         1743.5666374716,
                                         17.283094948715,
                                         0.1704403315308,
                                         0.0060871647822985,
                                         0.00095111888348471,
                                         0,
                                         0,
                                         90797.863755017,
                                         17.299608080995,
                                         0.17135102283357,
                                         0.001705973746997,
                                         6.0927621246887e-05,
                                         9.5199341224158e-06,
                                         0,
                                         0,
                                         90795.733286547,
                                         0.0017137623373227,
                                         1.7136869559277e-05,
                                         1.7061463826407e-07,
                                         6.0933785656387e-09,
                                         9.5208973051175e-10,
                                         0,
                                         0,
                                         90795.909178441,
                                         1.7137641049041e-07,
                                         1.7136886901341e-09,
                                         1.7061481088847e-11,
                                         6.093384730783e-13,
                                         9.5209069381481e-14,
                                         0,
                                         0};
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
