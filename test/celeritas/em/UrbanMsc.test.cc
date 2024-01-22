//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/msc/UrbanMsc.hh"

#include <random>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/grid/Interpolator.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/em/UrbanMscParams.hh"
#include "celeritas/em/msc/detail/MscStepFromGeo.hh"
#include "celeritas/em/msc/detail/MscStepToGeo.hh"
#include "celeritas/em/msc/detail/UrbanMscScatter.hh"
#include "celeritas/em/msc/detail/UrbanMscStepLimit.hh"
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
#include "celeritas/random/distribution/GenerateCanonical.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/SimParams.hh"
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
    UrbanPositronCorrector calc_h{1.0};  // Hydrogen
    UrbanPositronCorrector calc_w{74.0};  // Tungsten

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
// TEST HARNESS
//---------------------------------------------------------------------------//

class UrbanMscTest : public ::celeritas::test::RootTestBase
{
  protected:
    using RandomEngine = ::celeritas::test::DiagnosticRngEngine<std::mt19937>;

    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;
    using SimStateStore = CollectionStateStore<SimStateData, MemSpace::host>;
    using MevEnergy = units::MevEnergy;

    using Action = MscInteraction::Action;

  protected:
    std::string_view geometry_basename() const final
    {
        return "four-steel-slabs"sv;
    }

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
    ParticleTrackView make_par_view(PDGNumber pdg, MevEnergy energy)
    {
        CELER_EXPECT(pdg);
        CELER_EXPECT(energy > zero_quantity());
        auto pid = this->particle()->find(pdg);
        CELER_ASSERT(pid);

        ParticleTrackView par{this->particle()->host_ref(),
                              particle_state_.ref(),
                              TrackSlotId{0}};
        ParticleTrackView::Initializer_t init;
        init.particle_id = pid;
        init.energy = energy;
        par = init;
        return par;
    }

    PhysicsTrackView
    make_phys_view(ParticleTrackView const& par, std::string const& matname)
    {
        auto mid = this->material()->find_material(matname);
        CELER_ASSERT(mid);

        // Initialize physics
        PhysicsTrackView phys_view(this->physics()->host_ref(),
                                   physics_state_.ref(),
                                   par.particle_id(),
                                   mid,
                                   TrackSlotId{0});
        phys_view = PhysicsTrackInitializer{};

        // Calculate and store the energy loss (dedx) range limit
        auto ppid = phys_view.eloss_ppid();
        auto grid_id = phys_view.value_grid(ValueGridType::range, ppid);
        auto calc_range = phys_view.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(par.energy());
        phys_view.dedx_range(range);

        return phys_view;
    }

    GeoTrackView make_geo_view(real_type r)
    {
        GeoTrackView geo_view(
            this->geometry()->host_ref(), geo_state_.ref(), TrackSlotId{0});
        geo_view = {{r, r, r}, Real3{0, 0, 1}};
        return geo_view;
    }

  protected:
    std::shared_ptr<UrbanMscParams const> msc_params_;

    PhysicsStateStore physics_state_;
    ParticleStateStore particle_state_;
    GeoStateStore geo_state_;
    SimStateStore sim_state_;
};

struct PrintableParticle
{
    ParticleTrackView const& par;
    ParticleParams const& params;
};

std::ostream& operator<<(std::ostream& os, PrintableParticle const& pp)
{
    os << pp.params.id_to_label(pp.par.particle_id()) << " at "
       << value_as<units::MevEnergy>(pp.par.energy()) << " MeV";
    return os;
}

template<class T>
struct LabeledValue
{
    char const* label;
    T value;
};

// CTAD
template<typename T>
LabeledValue(char const*, T) -> LabeledValue<T>;

template<class T>
std::ostream& operator<<(std::ostream& os, LabeledValue<T> const& lv)
{
    os << lv.label << "=" << lv.value;
    return os;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UrbanMscTest, coeff_data)
{
    auto const& params = msc_params_->host_ref();

    EXPECT_SOFT_EQ(1e-4, value_as<MevEnergy>(params.params.low_energy_limit));
    EXPECT_SOFT_EQ(1e2, value_as<MevEnergy>(params.params.high_energy_limit));
    {
        // Check steel material data
        auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
        ASSERT_TRUE(mid);
        UrbanMscMaterialData const& md = params.material_data[mid];
        EXPECT_SOFT_EQ(md.stepmin_coeff[0], 1e3 * 4.4449610414595817);
        EXPECT_SOFT_EQ(md.stepmin_coeff[1], 1e3 * 1.5922149179564158);
        EXPECT_SOFT_EQ(md.theta_coeff[0], 0.97326969977637379);
        EXPECT_SOFT_EQ(md.theta_coeff[1], 0.044188139325421663);
        EXPECT_SOFT_EQ(md.tail_coeff[0], 1.6889578380303167);
        EXPECT_SOFT_EQ(md.tail_coeff[1], 2.745018223507488);
        EXPECT_SOFT_EQ(md.tail_coeff[2], -2.2531516772497562);
        EXPECT_SOFT_EQ(md.tail_corr, 0.052696806851297018);
    }

    // Check data for electron in stainless steel
    auto mid = this->material()->find_material("G4_STAINLESS-STEEL");
    ASSERT_TRUE(mid);
    auto pid = this->particle()->find(pdg::electron());
    ASSERT_TRUE(pid);
    UrbanMscParMatData const& par = params.par_mat_data[params.at(mid, pid)];
    EXPECT_SOFT_EQ(par.d_over_r, 0.64474963087322135);
}

TEST_F(UrbanMscTest, helper)
{
    auto par = this->make_par_view(pdg::electron(), MevEnergy{10.01});
    auto phys = this->make_phys_view(par, "G4_STAINLESS-STEEL");
    UrbanMscHelper helper(msc_params_->host_ref(), par, phys);

    EXPECT_SOFT_EQ(0.90681578657668238, phys.dedx_range());
    EXPECT_SOFT_EQ(1.0897296072933604, helper.calc_msc_mfp(MevEnergy{10.01}));
    EXPECT_SOFT_EQ(0.90820266262324023, helper.calc_msc_mfp(MevEnergy{9.01}));
    EXPECT_SOFT_EQ(11.039692548085707,
                   value_as<MevEnergy>(helper.calc_inverse_range(1.0)));
    EXPECT_SOFT_EQ(4.5491422239586035,
                   value_as<MevEnergy>(helper.calc_end_energy(0.5)));
}

TEST_F(UrbanMscTest, step_conversion)
{
    using LogInterp = Interpolator<Interp::linear, Interp::log, real_type>;
    constexpr int pstep_points = 8;
    constexpr int gstep_points = 8;

    UrbanMscParameters const& params = msc_params_->host_ref().params;

    auto test_one = [&](char const* mat, PDGNumber ptype, MevEnergy energy) {
        auto par = this->make_par_view(ptype, energy);
        auto phys = this->make_phys_view(par, mat);
        SCOPED_TRACE((PrintableParticle{par, *this->particle()}));
        UrbanMscHelper helper(msc_params_->host_ref(), par, phys);

        real_type range = phys.dedx_range();
        real_type lambda = helper.calc_msc_mfp(energy);
        MscStepToGeo calc_geom_path(
            msc_params_->host_ref(), helper, energy, lambda, range);

        LogInterp calc_pstep({0, real_type{0.9} * params.limit_min_fix()},
                             {static_cast<real_type>(pstep_points), range});
        for (auto ppt : celeritas::range(pstep_points + 1))
        {
            // Calculate given a physics step between "tiny" and the
            // maximum range
            real_type pstep = calc_pstep(ppt);
            if (ppt == pstep_points)
                pstep = range;

            SCOPED_TRACE((LabeledValue{"pstep", pstep}));
            // Get the equivalent "geometrical" step
            MscStepToGeo::result_type gp;
            ASSERT_NO_THROW(gp = calc_geom_path(pstep));
            EXPECT_LE(gp.step, pstep);
            EXPECT_GT(gp.step, 0);

            MscStep msc_step;
            msc_step.true_path = pstep;
            msc_step.geom_path = gp.step;
            msc_step.alpha = gp.alpha;
            MscStepFromGeo geo_to_true(
                msc_params_->host_ref().params, msc_step, range, lambda);
            LogInterp calc_gstep(
                {0, real_type{0.9} * params.limit_min_fix()},
                {static_cast<real_type>(gstep_points), gp.step});
            for (auto gpt : celeritas::range(gstep_points + 1))
            {
                // Calculate between a nearby hypothetical geometric
                // boundary and "no boundary" (i.e. pstep limited)
                real_type gstep = celeritas::min(calc_gstep(gpt), pstep);
                SCOPED_TRACE((LabeledValue{"gstep", gstep}));
                real_type true_step;
                ASSERT_NO_THROW(true_step = geo_to_true(gstep));
                EXPECT_LE(true_step, pstep);
                EXPECT_GE(true_step, gstep)
                    << LabeledValue{"true_step", true_step};
            }

            // Test exact true -> geo -> true conversion
            {
                real_type true_step{-1};
                ASSERT_NO_THROW(true_step = geo_to_true(gp.step));
                /*
                 * TODO: large relative error -0.00081720192362734587 when
                 pstep
                 * is near or equal to range:
                 *
                 z -> g: Low energy or range-limited step:
                    slope = 1.6653345369377e-15
                 g -> z: Exact inverse:
                   x = 1 = 1 - 1.1102230246252e-16,
                   w = 5.16719, alpha = 359.80425185237
                   => 0.99918279807637 / alpha
                 true_step=0.0027770177615531158
                 pstep=0.0027792890018717618
                 e- at 0.102364 MeV
                 */
                real_type tol = 1 - gp.alpha * pstep < 1e-8 ? 1e-3 : 1e-10;
                if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
                {
                    tol = std::sqrt(tol);
                }
                EXPECT_SOFT_NEAR(pstep, true_step, tol);
            }
        }
    };

    for (char const* mat : {"G4_STAINLESS-STEEL", "G4_Galactic"})
    {
        for (auto ptype : {pdg::electron(), pdg::positron()})
        {
            for (real_type energy : {99.999,
                                     51.0231,
                                     10.0564,
                                     5.05808,
                                     1.01162,
                                     0.501328,
                                     0.102364,
                                     0.0465336,
                                     0.00708839,
                                     1e-5})
            {
                test_one(mat, ptype, MevEnergy{energy});
            }
        }
    }
}

TEST_F(UrbanMscTest, TEST_IF_CELERITAS_DOUBLE(msc_scattering))
{
    // Test energies
    static real_type const energy[] = {51.0231,
                                       10.0564,
                                       5.05808,
                                       1.01162,
                                       0.501328,
                                       0.102364,
                                       0.0465336,
                                       0.00708839};
    constexpr auto nsamples = std::size(energy);

    // Calculate range instead of hardcoding to ensure step and range values
    // are bit-for-bit identical when range limits the step. The first three
    // steps are not limited by range
    constexpr double step_is_range = -1;
    std::vector<double> step = {0.00279169, 0.412343, 0.0376414};
    step.resize(nsamples, step_is_range);

    ASSERT_EQ(nsamples, step.size());

    {
        SimTrackView sim_track_view(
            this->sim()->host_ref(), sim_state_.ref(), TrackSlotId{0});
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
    std::vector<double> angle;
    std::vector<double> displace;
    std::vector<char> action;

    // Total RNG count (we only sample once per particle/energy so count and
    // average are the same)
    std::vector<int> avg_engine_samples;

    auto const& msc_params = msc_params_->host_ref();

    auto sample_one = [&](PDGNumber ptype, int i) {
        auto par = this->make_par_view(ptype, MevEnergy{energy[i]});
        auto phys = this->make_phys_view(par, "G4_STAINLESS-STEEL");
        auto geo = this->make_geo_view(/* r = */ i * 2 - real_type(1e-4));
        MaterialView mat = this->material()->get(phys.material_id());
        rng.reset_count();

        UrbanMscHelper helper(msc_params, par, phys);
        range.push_back(phys.dedx_range());
        lambda.push_back(helper.msc_mfp());

        real_type this_pstep = step[i];
        if (this_pstep == step_is_range)
        {
            this_pstep = phys.dedx_range();
        }
        pstep.push_back(this_pstep);

        // Calculate physical step limit due to MSC
        real_type safety = geo.find_safety();
        auto [true_path, displaced] = [&]() -> std::pair<real_type, bool> {
            if (this_pstep < msc_params.params.limit_min_fix()
                || safety >= helper.max_step())
            {
                // Small step or far from boundary
                return {this_pstep, false};
            }
            UrbanMscStepLimit calc_limit(msc_params,
                                         helper,
                                         par.energy(),
                                         &phys,
                                         mat.material_id(),
                                         geo.is_on_boundary(),
                                         safety,
                                         this_pstep);

            return {calc_limit(rng), true};
        }();
        tstep.push_back(true_path);

        // Convert physical step limit to geometrical step
        MscStepToGeo calc_geom_path(msc_params,
                                    helper,
                                    par.energy(),
                                    helper.msc_mfp(),
                                    phys.dedx_range());
        auto gp = calc_geom_path(true_path);
        gstep.push_back(gp.step);
        alpha.push_back(gp.alpha);

        MscStep step_result;
        step_result.true_path = true_path;
        step_result.geom_path = gp.step;
        step_result.alpha = gp.alpha;
        step_result.is_displaced = displaced;

        // No geo->phys conversion needed because we don't test for the
        // geo-limited case here (see the geo->true tests above)
        UrbanMscScatter scatter(
            msc_params, helper, par, phys, mat, geo.dir(), safety, step_result);
        MscInteraction sample_result = scatter(rng);

        angle.push_back(sample_result.action != Action::unchanged
                            ? sample_result.direction[0]
                            : 0);
        displace.push_back(sample_result.action == Action::displaced
                               ? sample_result.displacement[0]
                               : 0);
        action.push_back(sample_result.action == Action::displaced   ? 'd'
                         : sample_result.action == Action::scattered ? 's'
                                                                     : 'u');
        avg_engine_samples.push_back(rng.count());
    };

    for (auto ptype : {pdg::electron(), pdg::positron()})
    {
        for (auto i : celeritas::range(nsamples))
        {
            sample_one(ptype, i);
            if (i == 1 || i == 9)
            {
                // Test original RNG stream
                celeritas::generate_canonical(rng);
            }
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
        3.4500251375335, 12.793635146437, 0, 0, 1347.4250715939,
        32089.171921536, 0, 1.7061753085921, 3.3439279763905, 12.693879333563,
        0, 0, 1479.3077264327, 38389.697977001};
    static double const expected_angle[] = {0.00031474130607916,
        -0.0179600098014372, -0.14560882721751, 0, -0.32650640360665,
        0.013072020086723, 0, 0, 0.003112817663327, 0.385707466417037,
        0.17523769034715, 0, -0.30604942826098, 0.40930643323792, 0, 0};
    static double const expected_displace[] = {8.1986203515053e-06,
        9.86121281691721e-05, -7.1670542039709e-05, 0, -9.1137823713002e-05,
        9.7878389032256e-06, 0, 0, 5.3169211357544e-06, 3.04785478410349e-05,
        9.8992726876372e-05, 0, -9.0024133671603e-05, 2.9542258777685e-05, 0,
        0};
    static char const expected_action[] = {'d', 'd', 'd', 'u', 'd', 'd', 'u',
        'u', 'd', 'd', 'd', 'u', 'd', 'd', 'u', 'u'};
    static int const expected_avg_engine_samples[] = {12, 14, 16, 0, 16, 16, 0,
        0, 12, 14, 16, 0, 16, 16, 0, 0};
    // clang-format on

    EXPECT_VEC_SOFT_EQ(expected_pstep, pstep);
    EXPECT_VEC_SOFT_EQ(expected_range, range);
    EXPECT_VEC_SOFT_EQ(expected_lambda, lambda);
    EXPECT_VEC_SOFT_EQ(expected_tstep, tstep);
    EXPECT_VEC_SOFT_EQ(expected_gstep, gstep);
    EXPECT_VEC_SOFT_EQ(expected_alpha, alpha);
    EXPECT_VEC_NEAR(expected_angle, angle, 2e-12);
    EXPECT_VEC_NEAR(expected_displace, displace, 1e-11);
    EXPECT_VEC_EQ(expected_action, action);
    EXPECT_VEC_EQ(expected_avg_engine_samples, avg_engine_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
