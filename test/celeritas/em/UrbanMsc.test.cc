//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/msc/UrbanMsc.hh"

#include "corecel/cont/Range.hh"
#include "corecel/grid/Interpolator.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/em/msc/detail/MscStepFromGeo.hh"
#include "celeritas/em/msc/detail/MscStepToGeo.hh"
#include "celeritas/em/msc/detail/UrbanMscMinimalStepLimit.hh"
#include "celeritas/em/msc/detail/UrbanMscSafetyStepLimit.hh"
#include "celeritas/em/msc/detail/UrbanMscScatter.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/random/distribution/GenerateCanonical.hh"

#include "MscTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
struct InvCentimeter
{
    static CELER_CONSTEXPR_FUNCTION Constant value()
    {
        return 1 / units::centimeter;
    }
    static char const* label() { return "1/cm"; }
};

using InvCmAlpha = RealQuantity<InvCentimeter>;
using celeritas::test::from_cm;
using celeritas::test::to_cm;
using units::MevEnergy;

constexpr bool using_vecgeom_surface = CELERITAS_VECGEOM_SURFACE
                                       && CELERITAS_CORE_GEO
                                              == CELERITAS_CORE_GEO_VECGEOM;

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

class UrbanMscTest : public ::celeritas::test::MscTestBase
{
  protected:
    using Action = MscInteraction::Action;

    void SetUp() override
    {
        // Load MSC data
        msc_params_ = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());
        ASSERT_TRUE(msc_params_);
    }

  protected:
    std::shared_ptr<UrbanMscParams const> msc_params_;
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
    auto par = this->make_par_view(pdg::electron(), MevEnergy{10});
    auto phys = this->make_phys_view(
        par, "G4_STAINLESS-STEEL", this->physics()->host_ref());
    UrbanMscHelper helper(params, par, phys);
    EXPECT_SOFT_EQ(helper.pmdata().d_over_r, 0.64474963087322135);
}

TEST_F(UrbanMscTest, helper)
{
    auto par = this->make_par_view(pdg::electron(), MevEnergy{10.01});
    auto phys = this->make_phys_view(
        par, "G4_STAINLESS-STEEL", this->physics()->host_ref());
    UrbanMscHelper helper(msc_params_->host_ref(), par, phys);

    EXPECT_SOFT_EQ(0.90681578671073682, to_cm(phys.dedx_range()));
    EXPECT_SOFT_EQ(1.0897296072933604,
                   to_cm(helper.calc_msc_mfp(MevEnergy{10.01})));
    EXPECT_SOFT_EQ(0.90820266262324023,
                   to_cm(helper.calc_msc_mfp(MevEnergy{9.01})));
    EXPECT_SOFT_EQ(
        11.039692546604346,
        value_as<MevEnergy>(helper.calc_inverse_range(from_cm(1.0))));
    EXPECT_SOFT_EQ(4.5491422239586035,
                   value_as<MevEnergy>(helper.calc_end_energy(from_cm(0.5))));
}

TEST_F(UrbanMscTest, step_conversion)
{
    using LogInterp = Interpolator<Interp::linear, Interp::log, real_type>;
    constexpr int pstep_points = 8;
    constexpr int gstep_points = 8;

    UrbanMscParameters const& params = msc_params_->host_ref().params;

    auto test_one = [&](char const* mat, PDGNumber ptype, MevEnergy energy) {
        auto par = this->make_par_view(ptype, energy);
        auto phys = this->make_phys_view(par, mat, this->physics()->host_ref());
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
                EXPECT_SOFT_NEAR(pstep, true_step, tol)
                    << "Geo step = " << repr(gp.step)
                    << ", alpha = " << repr(gp.alpha);
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

TEST_F(UrbanMscTest, TEST_IF_CELERITAS_DOUBLE(step_limit))
{
    using Algorithm = MscStepLimitAlgorithm;

    struct Result
    {
        using VecReal = std::vector<real_type>;

        VecReal mean_step;
        VecReal range_init;
        VecReal range_factor;
        VecReal limit_min;
    };

    auto sample = [&](Algorithm alg, bool on_boundary) {
        RandomEngine& rng = this->rng();
        Result result;

        real_type const num_samples = 100;
        real_type const safety = 0;

        auto const& msc_params = msc_params_->host_ref();
        auto phys_params = this->physics()->host_ref();
        if (alg == Algorithm::minimal)
        {
            phys_params.scalars.light.step_limit_algorithm = Algorithm::minimal;
            phys_params.scalars.light.range_factor = 0.2;
        }
        else if (alg == Algorithm::safety_plus)
        {
            phys_params.scalars.light.step_limit_algorithm
                = Algorithm::safety_plus;
        }

        for (real_type energy : {0.01, 0.1, 1.0, 10.0, 100.0})
        {
            auto par = this->make_par_view(pdg::electron(), MevEnergy{energy});
            auto phys
                = this->make_phys_view(par, "G4_STAINLESS-STEEL", phys_params);
            EXPECT_FALSE(phys.msc_range());
            UrbanMscHelper helper(msc_params, par, phys);

            real_type mean_step = 0;
            for (int i = 0; i < num_samples; ++i)
            {
                real_type step = phys.dedx_range();
                EXPECT_FALSE(step < msc_params.params.limit_min_fix());
                if (alg == Algorithm::minimal)
                {
                    // Minimal step limit algorithm
                    UrbanMscMinimalStepLimit calc_limit(
                        msc_params, helper, &phys, on_boundary, step);
                    mean_step += calc_limit(rng);
                }
                else
                {
                    // Safety/safety plus step limit algorithm
                    UrbanMscSafetyStepLimit calc_limit(msc_params,
                                                       helper,
                                                       par,
                                                       &phys,
                                                       phys.material_id(),
                                                       on_boundary,
                                                       safety,
                                                       step);
                    mean_step += calc_limit(rng);
                }
            }
            result.mean_step.push_back(to_cm(mean_step / num_samples));

            auto const& msc_range = phys.msc_range();
            result.range_init.push_back(to_cm(msc_range.range_init));
            result.range_factor.push_back(msc_range.range_factor);
            result.limit_min.push_back(to_cm(msc_range.limit_min));
        }
        return result;
    };

    {
        // "Minimal" algorithm, first step and not on boundary
        // step = phys_step
        static double const expected_mean_step[] = {5.4443542524765e-05,
                                                    0.0026634570717113,
                                                    0.07706894630323,
                                                    0.90591081547834,
                                                    8.8845468955896};
        static double const expected_range_init[] = {inf, inf, inf, inf, inf};
        static double const expected_range_factor[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        static double const expected_limit_min[]
            = {1e-08, 1e-08, 1e-08, 1e-08, 1e-08};

        auto result = sample(Algorithm::minimal, false);
        EXPECT_VEC_SOFT_EQ(expected_mean_step, result.mean_step);
        EXPECT_VEC_EQ(expected_range_init, result.range_init);
        EXPECT_VEC_EQ(expected_range_factor, result.range_factor);
        EXPECT_VEC_EQ(expected_limit_min, result.limit_min);
    }
    {
        // "Minimal" algorithm, first step and on boundary
        static double const expected_mean_step[] = {1.0947992605048e-05,
                                                    0.00052916041217278,
                                                    0.015353327646933,
                                                    0.21986948123587,
                                                    8.8845468955896};
        static double const expected_range_init[] = {1.0888708504953e-05,
                                                     0.00053269141434226,
                                                     0.015413789260646,
                                                     0.21762788543933,
                                                     15.553546812173};
        static double const expected_range_factor[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        static double const expected_limit_min[]
            = {1e-08, 1e-08, 1e-08, 1e-08, 1e-08};

        auto result = sample(Algorithm::minimal, true);
        EXPECT_VEC_SOFT_EQ(expected_mean_step, result.mean_step);
        EXPECT_VEC_SOFT_EQ(expected_range_init, result.range_init);
        EXPECT_VEC_EQ(expected_range_factor, result.range_factor);
        EXPECT_VEC_EQ(expected_limit_min, result.limit_min);
    }
    {
        // "Use safety" algorithm
        static double const expected_mean_step[] = {2.1774398678028e-06,
                                                    0.00010553667369532,
                                                    0.0030850804312135,
                                                    0.15209571378504,
                                                    8.8845468955896};
        static double const expected_range_init[] = {5.4443542524765e-05,
                                                     0.0026634570717113,
                                                     0.07706894630323,
                                                     1.0881394271966,
                                                     77.767734060865};
        static double const expected_range_factor[]
            = {0.04, 0.04, 0.04, 0.13881394271966, 7.8067734060865};
        static double const expected_limit_min[] = {1.9688399316472e-06,
                                                    1.0522532283188e-05,
                                                    3.1432398888924e-05,
                                                    4.0583539826243e-05,
                                                    3.6094312868035e-05};

        auto result = sample(Algorithm::safety, true);
        EXPECT_VEC_SOFT_EQ(expected_mean_step, result.mean_step);
        EXPECT_VEC_SOFT_EQ(expected_range_init, result.range_init);
        EXPECT_VEC_SOFT_EQ(expected_range_factor, result.range_factor);
        EXPECT_VEC_SOFT_EQ(expected_limit_min, result.limit_min);
    }
    {
        // "Use safety plus" algorithm
        static double const expected_mean_step[] = {2.1762412527058e-06,
                                                    0.00010784096017973,
                                                    0.0030577621335327,
                                                    0.094075756535309,
                                                    3.1108913402956};
        static double const expected_range_init[] = {5.4443542524765e-05,
                                                     0.0026634570717113,
                                                     0.07706894630323,
                                                     0.90591081547834,
                                                     8.8845468955896};
        static double const expected_range_factor[]
            = {0.04, 0.04, 0.04, 0.10324092334058, 5.0107349798953};
        static double const expected_limit_min[] = {1.9688399316472e-06,
                                                    1.0522532283188e-05,
                                                    3.1432398888924e-05,
                                                    4.0583539826243e-05,
                                                    3.6094312868035e-05};

        auto result = sample(Algorithm::safety_plus, true);
        EXPECT_VEC_SOFT_EQ(expected_mean_step, result.mean_step);
        EXPECT_VEC_SOFT_EQ(expected_range_init, result.range_init);
        EXPECT_VEC_SOFT_EQ(expected_range_factor, result.range_factor);
        EXPECT_VEC_SOFT_EQ(expected_limit_min, result.limit_min);
    }
}

constexpr double step_is_range = -1;

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
    std::vector<double> step = {0.00279169, 0.412343, 0.0376414};  // [cm]
    step.resize(nsamples, step_is_range);

    ASSERT_EQ(nsamples, step.size());

    RandomEngine& rng = this->rng();
    // Input and helper data
    std::vector<double> pstep;  // [cm]
    std::vector<double> range;  // [cm]
    std::vector<double> lambda;  // [cm]

    // Step limit
    std::vector<double> tstep;  // [cm]
    std::vector<double> gstep;  // [cm]
    std::vector<double> alpha;
    std::vector<double> msc_range_limit;  // [cm]

    // Scatter
    std::vector<double> angle;
    std::vector<double> displace;  // [cm]
    std::vector<char> action;

    // Total RNG count (we only sample once per particle/energy so count and
    // average are the same)
    std::vector<int> avg_engine_samples;

    auto const& msc_params = msc_params_->host_ref();

    auto sample_one = [&](PDGNumber ptype, int i) {
        real_type radius = from_cm(i * 2 - real_type(1e-4));

        auto par = this->make_par_view(ptype, MevEnergy{energy[i]});
        auto phys = this->make_phys_view(
            par, "G4_STAINLESS-STEEL", this->physics()->host_ref());
        auto geo = this->make_geo_view(radius);
        MaterialView mat = this->material()->get(phys.material_id());
        rng = this->rng();

        UrbanMscHelper helper(msc_params, par, phys);
        range.push_back(to_cm(phys.dedx_range()));
        lambda.push_back(to_cm(helper.msc_mfp()));

        real_type const this_pstep = [i, &phys, &step] {
            if (step[i] == step_is_range)
            {
                return phys.dedx_range();
            }
            real_type const pstep = from_cm(step[i]);
            CELER_VALIDATE(pstep <= phys.dedx_range(),
                           << "unit test input pstep=" << pstep
                           << " exceeds physics range " << phys.dedx_range());
            return pstep;
        }();
        pstep.push_back(to_cm(this_pstep));

        // Calculate physical step limit due to MSC
        real_type safety = geo.find_safety();
        auto [true_path, displaced] = [&]() -> std::pair<real_type, bool> {
            EXPECT_FALSE(phys.msc_range());
            if (this_pstep < msc_params.params.limit_min_fix()
                || safety >= helper.max_step())
            {
                // Small step or far from boundary
                msc_range_limit.push_back(-1);
                return {this_pstep, false};
            }
            UrbanMscSafetyStepLimit calc_limit(msc_params,
                                               helper,
                                               par,
                                               &phys,
                                               mat.material_id(),
                                               geo.is_on_boundary(),
                                               safety,
                                               this_pstep);

            // MSC range should be updated during construction of the limit
            // calculator
            msc_range_limit.push_back(to_cm(phys.msc_range().limit_min));

            return {calc_limit(rng), true};
        }();
        tstep.push_back(to_cm(true_path));

        // Convert physical step limit to geometrical step
        MscStepToGeo calc_geom_path(msc_params,
                                    helper,
                                    par.energy(),
                                    helper.msc_mfp(),
                                    phys.dedx_range());
        auto gp = calc_geom_path(true_path);
        gstep.push_back(to_cm(gp.step));
        alpha.push_back(native_value_to<InvCmAlpha>(gp.alpha).value());

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
                               ? to_cm(sample_result.displacement[0])
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

    static double const expected_pstep[] = {
        0.00279169,
        0.412343,
        0.0376414,
        0.07816386744463,
        0.031624623366637,
        0.0027792891324765,
        0.0007421564286833,
        3.1163288786189e-05,
        0.00279169,
        0.412343,
        0.0376414,
        0.078778123988169,
        0.031303585136774,
        0.0026015196588089,
        0.00067599187481117,
        2.6048657211831e-05,
    };
    static double const expected_range[] = {
        4.5639217208474,
        0.91101485322907,
        0.45387592065389,
        0.07816386744463,
        0.031624623366637,
        0.0027792891324765,
        0.0007421564286833,
        3.1163288786189e-05,
        4.6052228038519,
        0.93344757374528,
        0.46823131622043,
        0.078778123988169,
        0.031303585136774,
        0.0026015196588089,
        0.00067599187481117,
        2.6048657211831e-05,
    };
    static double const expected_lambda[] = {
        20.538835907703,
        1.0971140774006,
        0.33351871980427,
        0.025445778924487,
        0.0087509389402756,
        0.00066694512451,
        0.00017137575823624,
        6.8484179743242e-06,
        20.538835907703,
        1.2024899663097,
        0.36938283004206,
        0.028834889384336,
        0.0099285992018056,
        0.00075519594407854,
        0.0001990403696419,
        9.8568978271595e-06,
    };
    static double const expected_tstep[] = {
        0.00279169,
        0.15497550035228,
        0.0376414,
        0.07816386744463,
        0.0013704878271868,
        9.6599315266596e-05,
        0.0007421564286833,
        3.1163288786189e-05,
        0.00279169,
        0.19292164062171,
        0.028493079925032,
        0.078778123988169,
        0.0011819750905666,
        0.00011040360882338,
        0.00067599187481117,
        2.6048657211831e-05,
    };
    static double const expected_gstep[] = {
        0.0027915002818486,
        0.14348626259532,
        0.035504762192774,
        0.019196479870158,
        0.001268561154396,
        8.9929529063304e-05,
        0.00013922620627564,
        5.6145657548872e-06,
        0.0027915002818486,
        0.17644490740217,
        0.027387302420224,
        0.021108585476009,
        0.0011143302247454,
        0.00010271284328128,
        0.00015376538806457,
        7.1509534497628e-06,
    };
    static double const expected_alpha[] = {
        0,
        1.7708716862051,
        3.450025137536,
        12.793635124418,
        0,
        0,
        1347.4248303342,
        32089.039345654,
        0,
        1.7061753085921,
        3.3439279763904,
        12.693879333179,
        0,
        0,
        1479.3077213825,
        38389.694788022,
    };
    static double const expected_msc_range_limit[] = {
        3.5686548881735e-05,
        4.0510166112733e-05,
        4.0073894011965e-05,
        -1,
        2.5270068437103e-05,
        1.0695397351015e-05,
        -1,
        -1,
        1.6703727150203e-05,
        2.0782727059196e-05,
        2.0774322573966e-05,
        -1,
        1.3419878391705e-05,
        5.6685934033174e-06,
        -1,
        -1,
    };
    static double const expected_angle[] = {
        0.00031474130607916,
        -0.017960009801437,
        -0.14560882721752,
        0,
        -0.32650640437126,
        0.01307202043885,
        0,
        0,
        0.003112817663327,
        0.38570746641704,
        0.17523769034766,
        0,
        -0.30604942827394,
        0.40930643343077,
        0,
        0,
    };
    static double const expected_displace[] = {
        8.1986203515053e-06,
        9.8612128169172e-05,
        -7.1670542039709e-05,
        0,
        -9.1137823713002e-05,
        9.7878395637005e-06,
        0,
        0,
        5.3169211357544e-06,
        3.0478547841035e-05,
        9.8992726876372e-05,
        0,
        -9.0024133671603e-05,
        2.9542258814356e-05,
        0,
        0,
    };
    static char const expected_action[] = {
        'd',
        'd',
        'd',
        'u',
        'd',
        'd',
        'u',
        'u',
        'd',
        'd',
        'd',
        'u',
        'd',
        'd',
        'u',
        'u',
    };
    static int const expected_avg_engine_samples[] = {
        12,
        14,
        16,
        0,
        16,
        16,
        0,
        0,
        12,
        14,
        16,
        0,
        16,
        16,
        0,
        0,
    };

    EXPECT_VEC_SOFT_EQ(expected_pstep, pstep);
    EXPECT_VEC_SOFT_EQ(expected_range, range);
    EXPECT_VEC_SOFT_EQ(expected_lambda, lambda);
    EXPECT_VEC_SOFT_EQ(expected_tstep, tstep);
    EXPECT_VEC_SOFT_EQ(expected_gstep, gstep);
    EXPECT_VEC_SOFT_EQ(expected_alpha, alpha);
    EXPECT_VEC_SOFT_EQ(expected_msc_range_limit, msc_range_limit);
    EXPECT_VEC_NEAR(expected_angle, angle, 2e-12);
    EXPECT_VEC_NEAR(
        expected_displace, displace, using_vecgeom_surface ? 1e-2 : 1e-12);
    EXPECT_VEC_EQ(expected_action, action);
    EXPECT_VEC_EQ(expected_avg_engine_samples, avg_engine_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
