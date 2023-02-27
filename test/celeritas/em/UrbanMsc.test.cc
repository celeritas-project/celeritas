//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/msc/UrbanMsc.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/em/msc/UrbanMscScatter.hh"
#include "celeritas/em/msc/UrbanMscStepLimit.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/SimTrackView.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(UrbanPositronCorrectorTest, all)
{
    UrbanPositronCorrector calc_h{1.0};
    UrbanPositronCorrector calc_w{74.0};

    std::vector<real_type> actual_h;
    std::vector<real_type> actual_w;
    for (real_type y :
         {1e-3, 1e-2, 0.4, 0.5, 0.6, 1.0, 1.5, 2., 10., 1e2, 1e3, 1e6})
    {
        actual_h.push_back(calc_h(y));
        actual_w.push_back(calc_w(y));
    }

    // clang-format off
    static const double expected_h[] = {1.378751990475, 1.3787519983432,
        1.3813527280086, 1.3825378340463, 1.3834564182635, 1.3856807011387,
        1.3865656925136, 1.3865681880571, 1.3876210627429, 1.3882415266217,
        1.3882507402225, 1.3882508352478};
    static const double expected_w[] = {0.21482671339734,
        0.4833017838367, 0.70738388881252, 0.70471228941815, 0.7026415135041,
        0.69762728474033, 0.69563878645763, 0.69577660924627, 0.75392431413533,
        0.78819102317998, 0.78869986791365, 0.78870511592834};
    // clang-format on
    EXPECT_VEC_SOFT_EQ(expected_h, actual_h);
    EXPECT_VEC_SOFT_EQ(expected_w, actual_w);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail

namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public RootTestBase
{
  protected:
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;
    using SimStateStore = CollectionStateStore<SimStateData, MemSpace::host>;
    using MevEnergy = units::MevEnergy;

    using Action = MscInteraction::Action;

  protected:
    char const* geometry_basename() const final { return "four-steel-slabs"; }

    void SetUp() override
    {
        // Load MSC data
        msc_params_ = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        ASSERT_TRUE(msc_params_);

        // Allocate particle state
        auto state_size = 1;
        physics_state_
            = PhysicsStateStore(this->physics()->host_ref(), state_size);
        particle_state_
            = ParticleStateStore(this->particle()->host_ref(), state_size);
        geo_state_ = GeoStateStore(this->geometry()->host_ref(), state_size);
        sim_state_ = SimStateStore(state_size);
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    // Initialize a track
    PhysicsTrackView
    make_track_view(PDGNumber pdg, MaterialId mid, MevEnergy energy)
    {
        CELER_EXPECT(pdg);
        CELER_EXPECT(mid);
        CELER_EXPECT(energy > zero_quantity());

        auto pid = this->particle()->find(pdg);
        CELER_ASSERT(pid);
        const ThreadId tid{0};

        // Initialize particle
        {
            ParticleTrackView par{
                this->particle()->host_ref(), particle_state_.ref(), tid};
            ParticleTrackView::Initializer_t init;
            init.particle_id = pid;
            init.energy = energy;
            par = init;
        }

        // Initialize physics
        PhysicsTrackView phys_view(
            this->physics()->host_ref(), physics_state_.ref(), pid, mid, tid);
        phys_view = PhysicsTrackInitializer{};

        // Calculate and store the energy loss (dedx) range limit
        auto ppid = phys_view.eloss_ppid();
        auto grid_id = phys_view.value_grid(ValueGridType::range, ppid);
        auto calc_range = phys_view.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(energy);
        phys_view.dedx_range(range);

        return phys_view;
    }

  protected:
    std::shared_ptr<UrbanMscParams const> msc_params_;

    PhysicsStateStore physics_state_;
    ParticleStateStore particle_state_;
    GeoStateStore geo_state_;
    SimStateStore sim_state_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UrbanMscTest, coeff_data)
{
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    ASSERT_TRUE(mid);

    // Check MscMaterialDara for the current material (G4_STAINLESS-STEEL)
    UrbanMscMaterialData const& msc
        = msc_params_->host_ref().material_data[mid];

    EXPECT_SOFT_EQ(msc.coeffth1, 0.97326969977637379);
    EXPECT_SOFT_EQ(msc.coeffth2, 0.044188139325421663);
    EXPECT_SOFT_EQ(msc.d[0], 1.6889578380303167);
    EXPECT_SOFT_EQ(msc.d[1], 2.745018223507488);
    EXPECT_SOFT_EQ(msc.d[2], -2.2531516772497562);
    EXPECT_SOFT_EQ(msc.d[3], 0.052696806851297018);
    EXPECT_SOFT_EQ(msc.stepmin_a, 1e3 * 4.4449610414595817);
    EXPECT_SOFT_EQ(msc.stepmin_b, 1e3 * 1.5922149179564158);
    EXPECT_SOFT_EQ(msc.d_over_r, 0.64474963087322135);
    EXPECT_SOFT_EQ(msc.d_over_r_mh, 1.1248191999999999);
}

TEST_F(UrbanMscTest, helper)
{
    ParticleTrackView par{
        this->particle()->host_ref(), particle_state_.ref(), ThreadId{0}};
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    PhysicsTrackView phys
        = this->make_track_view(pdg::electron(), mid, MevEnergy{10.01});
    UrbanMscHelper helper(msc_params_->host_ref(), par, phys);

    EXPECT_SOFT_EQ(0.90681578657668238, phys.dedx_range());
    EXPECT_SOFT_EQ(1.0897296072933604, helper.calc_msc_mfp(MevEnergy{10.01}));
    EXPECT_SOFT_EQ(0.90820266262324023, helper.calc_msc_mfp(MevEnergy{9.01}));
    EXPECT_SOFT_EQ(11.039692548085707,
                   value_as<MevEnergy>(helper.calc_stopping_energy(1.0)));
    EXPECT_SOFT_EQ(4.5491422239586035,
                   value_as<MevEnergy>(helper.calc_end_energy(0.5)));
}

TEST_F(UrbanMscTest, msc_scattering)
{
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    ASSERT_TRUE(mid);

    // Test the step limitation algorithm and the msc sample scattering
    MscStep step_result;
    MscInteraction sample_result;

    // Input
    static const real_type energy[] = {51.0231,
                                       10.0564,
                                       5.05808,
                                       1.01162,
                                       0.501328,
                                       0.102364,
                                       0.0465336,
                                       0.00708839};
    constexpr unsigned int nsamples = std::end(energy) - std::begin(energy);

    // Calculate range instead of hardcoding to ensure step and range values
    // are bit-for-bit identical when range limits the step. The first three
    // steps are not limited by range
    constexpr double step_is_range = -1;
    std::vector<double> step = {0.00279169, 0.412343, 0.0376414};
    step.resize(nsamples, step_is_range);

    ASSERT_EQ(nsamples, step.size());

    {
        SimTrackView sim_track_view(sim_state_.ref(), ThreadId{0});
        sim_track_view = {};
        EXPECT_EQ(0, sim_track_view.num_steps());
    }

    RandomEngine rng;
    // Input and helper data
    std::vector<double> pstep;
    std::vector<double> range;
    std::vector<double> lambda;

    // Step limit
    std::vector<double> tstep;
    std::vector<double> gstep;
    std::vector<double> alpha;

    // Scatter
    std::vector<double> fstep;
    std::vector<double> angle;
    std::vector<double> displace;
    std::vector<char> action;

    auto sample_one = [&](PDGNumber ptype, int i) {
        ParticleTrackView par{
            this->particle()->host_ref(), particle_state_.ref(), ThreadId{0}};
        GeoTrackView geo_view(
            this->geometry()->host_ref(), geo_state_.ref(), ThreadId{0});
        MaterialView material_view = this->material()->get(mid);

        real_type r = i * 2 - real_type(1e-4);
        geo_view = {{r, r, r}, Real3{0, 0, 1}};

        MevEnergy inc_energy = MevEnergy{energy[i]};
        PhysicsTrackView phys = this->make_track_view(ptype, mid, inc_energy);

        UrbanMscHelper helper(msc_params_->host_ref(), par, phys);
        range.push_back(phys.dedx_range());
        lambda.push_back(helper.calc_msc_mfp(inc_energy));

        real_type this_pstep = step[i];
        if (this_pstep == step_is_range)
        {
            PhysicsTrackView phys
                = this->make_track_view(ptype, mid, MevEnergy{energy[i]});
            this_pstep = phys.dedx_range();
        }
        pstep.push_back(this_pstep);

        UrbanMscStepLimit calc_limit(msc_params_->host_ref(),
                                     par,
                                     &phys,
                                     material_view.material_id(),
                                     geo_view.is_on_boundary(),
                                     geo_view.find_safety(),
                                     this_pstep);

        step_result = calc_limit(rng);
        tstep.push_back(step_result.true_path);
        gstep.push_back(step_result.geom_path);
        alpha.push_back(step_result.alpha);

        UrbanMscScatter scatter(msc_params_->host_ref(),
                                par,
                                &geo_view,
                                phys,
                                material_view,
                                step_result,
                                this_pstep,
                                /* geo_limited = */ false);

        sample_result = scatter(rng);

        fstep.push_back(sample_result.step_length);
        angle.push_back(sample_result.action != Action::unchanged
                            ? sample_result.direction[0]
                            : 0);
        displace.push_back(sample_result.action == Action::displaced
                               ? sample_result.displacement[0]
                               : 0);
        action.push_back(sample_result.action == Action::displaced   ? 'd'
                         : sample_result.action == Action::scattered ? 's'
                                                                     : 'u');
    };

    for (auto ptype : {pdg::electron(), pdg::positron()})
    {
        for (auto i : celeritas::range(nsamples))
        {
            sample_one(ptype, i);
        }
    }

    // clang-format off
    static double const expected_pstep[] = {0.00279169, 0.412343, 0.0376414,
        0.078163867310103, 0.031624623231734, 0.0027792890018718,
        0.00074215629579836, 3.1163160035578e-05, 0.00279169, 0.412343,
        0.0376414, 0.078778123985786, 0.031303585134373, 0.0026015196566029,
        0.00067599187250339, 2.6048655048005e-05};
    static double const expected_range[] = {4.5639217207134, 0.91101485309501,
        0.45387592051985, 0.078163867310103, 0.031624623231734,
        0.0027792890018718, 0.00074215629579836, 3.1163160035578e-05,
        4.6052228038495, 0.93344757374292, 0.46823131621808, 0.078778123985786,
        0.031303585134373, 0.0026015196566029, 0.00067599187250339,
        2.6048655048005e-05};
    static double const expected_lambda[] = {20.538835907703, 1.0971140774006,
        0.33351871980427, 0.025445778924487, 0.0087509389402756,
        0.00066694512451, 0.00017137575823624, 6.8484179743242e-06,
        20.538835907703, 1.2024899663097, 0.36938283004206, 0.028834889384336,
        0.0099285992018056, 0.00075519594407854, 0.0001990403696419,
        9.8568978271595e-06};
    static double const expected_tstep[] = {0.00279169, 0.15497550035228,
        0.0376414, 0.078163867310103, 0.0013704878213315, 9.659931080008e-05,
        0.00074215629579836, 3.1163160035578e-05, 0.00279169, 0.19292164062171,
        0.028493079924889, 0.078778123985786, 0.001181975090476,
        0.00011040360872946, 0.00067599187250339, 2.6048655048005e-05};
    static double const expected_gstep[] = {0.0027915002818486,
        0.14348626259532, 0.035504762192774, 0.019196479862044,
        0.0012685611493895, 8.9929525199044e-05, 0.00013922620159908,
        5.6145615756546e-06, 0.0027915002818486, 0.17644490740217,
        0.027387302420092, 0.021108585475838, 0.001114330224665,
        0.00010271284320013, 0.00015376538794516, 7.1509532866908e-06};
    static double const expected_alpha[] = {0, 1.7708716862049,
        3.4500251375335, 0, 0, 0, 0, 0, 0, 1.7061753085921, 3.3439279763905, 0,
        0, 0, 0, 0};
    static double const expected_fstep[] = {0.00279169, 0.15497550035228,
        0.0376414, 0.078163867310103, 0.0013704878213315, 9.659931080008e-05,
        0.00074215629579836, 3.1163160035578e-05, 0.00279169, 0.19292164062171,
        0.028493079924889, 0.078778123985786, 0.001181975090476,
        0.00011040360872946, 0.00067599187250339, 2.6048655048005e-05};
    static double const expected_angle[] = {0.00031474130607916,
        0.79003683103898, -0.14560882721751, 0, -0.32650640360665,
        0.013072020086723, 0, 0, 0.003112817663327, 0.055689200859297,
        0.17523769034715, 0, -0.30604942826098, 0.40930643323792, 0, 0};
    static double const expected_displace[] = {8.1986203515053e-06,
        9.7530617641316e-05, -7.1670542039709e-05, 0, -9.1137823713002e-05,
        9.7878389032256e-06, 0, 0, 5.3169211357544e-06, 7.9159745553753e-05,
        9.8992726876372e-05, 0, -9.0024133671603e-05, 2.9542258777685e-05, 0,
        0};
    static char const expected_action[] = {'d', 'd', 'd', 'u', 'd', 'd', 'u',
        'u', 'd', 'd', 'd', 'u', 'd', 'd', 'u', 'u'};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_pstep, pstep);
    EXPECT_VEC_SOFT_EQ(expected_range, range);
    EXPECT_VEC_SOFT_EQ(expected_lambda, lambda);
    EXPECT_VEC_SOFT_EQ(expected_tstep, tstep);
    EXPECT_VEC_SOFT_EQ(expected_gstep, gstep);
    EXPECT_VEC_SOFT_EQ(expected_alpha, alpha);
    EXPECT_VEC_SOFT_EQ(expected_fstep, fstep);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);
    EXPECT_VEC_SOFT_EQ(expected_displace, displace);
    EXPECT_VEC_EQ(expected_action, action);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
