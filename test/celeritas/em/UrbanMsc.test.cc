//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMsc.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/em/msc/UrbanMsc.hh"

#include "corecel/cont/Range.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/Histogram.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
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

#include "MscTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
using celeritas::test::from_cm;
using celeritas::test::Histogram;
using celeritas::test::to_cm;
using units::MevEnergy;

constexpr bool using_vecgeom_surface = CELERITAS_VECGEOM_SURFACE
                                       && CELERITAS_CORE_GEO
                                              == CELERITAS_CORE_GEO_VECGEOM;

template<class T>
void extend_from_histogram(std::vector<T>& v, Histogram const& h)
{
    auto dens = h.calc_density();
    v.insert(v.end(), dens.begin(), dens.end());
}

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

#define UrbanMscTest TEST_IF_CELERITAS_USE_ROOT(UrbanMscTest)
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

// NOTE: these have to live outside the function due to bugs in MSVC's
// lambda handling
constexpr int num_samples = 10000;
constexpr real_type step_is_range = -1;

TEST_F(UrbanMscTest, msc_scattering)
{
    // Test energies (MeV)
    static real_type const energy[] = {
        51.0231,
        10.0564,
        10.0,
        5.05808,
        1.01162,
        1.01,
        1.01,
        0.501328,
        0.102364,
        0.0465336,
        0.00708839,
    };
    constexpr auto num_energy = std::size(energy);

    // Instead of hardcoding range, ensure step and range values
    // are bit-for-bit identical when range limits the step using a sentinel.
    std::vector<real_type> step = {
        0.00279169,
        0.412343,
        step_is_range,
        0.0376414,
        step_is_range,
        1e-4,
        1e-7,
    };
    step.resize(num_energy, step_is_range);
    ASSERT_EQ(num_energy, step.size());

    auto const get_pstep = [&step](int i, PhysicsTrackView const& phys) {
        CELER_EXPECT(static_cast<size_type>(i) < step.size());
        if (step[i] == step_is_range)
        {
            return phys.dedx_range();
        }
        real_type const pstep = from_cm(step[i]);
        CELER_VALIDATE(pstep <= phys.dedx_range(),
                       << "unit test input pstep=" << pstep
                       << " exceeds physics range " << phys.dedx_range());
        return pstep;
    };

    RandomEngine& rng = this->rng();

    // Input and helper data
    std::vector<real_type> lambda_cm;  // [cm]
    std::vector<real_type> pstep_mfp;
    std::vector<real_type> range_mfp;

    // Step limit
    std::vector<real_type> msc_range_mfp;
    std::vector<real_type> tstep_frac;  // fraction of physics step
    std::vector<real_type> gstep_frac;  // fraction of true path
    std::vector<real_type> alpha_over_mfp;

    // Binned scattering results
    std::vector<real_type> angle;
    std::vector<real_type> displace_frac;  // fraction of safety
    std::vector<real_type> action;

    // Average RNG per scatter
    std::vector<real_type> avg_engine_samples;

    auto const& msc_params = msc_params_->host_ref();

    auto run_particle = [&](PDGNumber ptype, int i, real_type safety_mfp) {
        // Set up state; geometry is arbitrary, direction is +z
        auto par = this->make_par_view(ptype, MevEnergy{energy[i]});
        auto phys = this->make_phys_view(
            par, "G4_STAINLESS-STEEL", this->physics()->host_ref());
        auto geo = this->make_geo_view(from_cm(i * 2 - real_type(1e-4)));
        MaterialView mat = this->material()->get(phys.material_id());
        real_type this_pstep = get_pstep(i, phys);

        // MSC parameters
        UrbanMscHelper helper(msc_params, par, phys);
        real_type const mfp = helper.msc_mfp();
        lambda_cm.push_back(to_cm(mfp));
        pstep_mfp.push_back(this_pstep / mfp);
        range_mfp.push_back(phys.dedx_range() / mfp);

        // Calculate physical step limit due to MSC
        real_type const safety = safety_mfp * mfp;
        real_type true_path;
        bool displaced;
        std::tie(true_path, displaced) = [&]() -> std::pair<real_type, bool> {
            EXPECT_FALSE(phys.msc_range());
            if (this_pstep < msc_params.params.limit_min_fix()
                || safety >= helper.max_step())
            {
                // Small step or far from boundary
                msc_range_mfp.push_back(-1);
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
            msc_range_mfp.push_back(phys.msc_range().limit_min / mfp);
            return {calc_limit(rng), true};
        }();
        tstep_frac.push_back(true_path / this_pstep);

        // Reset RNG from the one sample
        avg_engine_samples.push_back(static_cast<double>(rng.exchange_count()));

        // Convert physical step limit to geometrical step
        MscStep step_result = [&] {
            MscStepToGeo calc_geom_path(msc_params,
                                        helper,
                                        par.energy(),
                                        helper.msc_mfp(),
                                        phys.dedx_range());
            // Note: alpha units are 1/len, so we multiply by mfp
            auto gp = calc_geom_path(true_path);
            gstep_frac.push_back(gp.step / true_path);
            alpha_over_mfp.push_back(gp.alpha * mfp);
            MscStep result;
            result.true_path = true_path;
            result.geom_path = gp.step;
            result.alpha = gp.alpha;
            result.is_displaced = displaced;
            return result;
        }();

        // No geo->phys conversion needed because we don't test for the
        // geo-limited case here (see the geo->true tests above)
        UrbanMscScatter scatter(
            msc_params, helper, par, phys, mat, geo.dir(), safety, step_result);

        // Angle is the change in angle (original was +z)
        Histogram bin_angle(9, {-1, 1});
        // Fraction of safety
        real_type avg_displacement{0};
        // Interaction type: unchanged, scattered, displaced
        static_assert(static_cast<int>(MscInteraction::Action::size_) == 3);
        Histogram bin_action(3, {-0.5, 2.5});

        for (int i = 0; i < num_samples; ++i)
        {
            MscInteraction interaction = scatter(rng);

            // Bin change in angle from +z
            bin_angle(interaction.direction[2]);
            // Bin isotropic displacement distance as fraction of geo path
            avg_displacement += (interaction.action == Action::displaced
                                     ? norm(interaction.displacement) / safety
                                     : 0);
            bin_action(static_cast<real_type>(interaction.action));
        }

        extend_from_histogram(angle, bin_angle);
        displace_frac.push_back(avg_displacement / num_samples);
        extend_from_histogram(action, bin_action);
        avg_engine_samples.push_back(
            static_cast<real_type>(rng.exchange_count())
            / static_cast<real_type>(num_samples));
    };

    for (auto ptype : {pdg::electron(), pdg::positron()})
    {
        for (auto i : range(num_energy))
        {
            for (auto safety_mfp : {0.1, 1.0, 10.0})
            {
                run_particle(ptype, i, safety_mfp);
            }
        }
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
    {
        PRINT_EXPECTED(lambda_cm);
        PRINT_EXPECTED(pstep_mfp);
        PRINT_EXPECTED(range_mfp);

        PRINT_EXPECTED(msc_range_mfp);
        PRINT_EXPECTED(tstep_frac);
        PRINT_EXPECTED(gstep_frac);
        PRINT_EXPECTED(alpha_over_mfp);

        PRINT_EXPECTED(angle);
        PRINT_EXPECTED(displace_frac);
        PRINT_EXPECTED(action);

        PRINT_EXPECTED(avg_engine_samples);

        GTEST_SKIP() << "Not tested for single precision";
    }

    static double const expected_lambda_cm[] = {
        20.538835907703,     20.538835907703,     20.538835907703,
        1.0971140774006,     1.0971140774006,     1.0971140774006,
        1.0881394271966,     1.0881394271966,     1.0881394271966,
        0.33351871980427,    0.33351871980427,    0.33351871980427,
        0.025445778924487,   0.025445778924487,   0.025445778924487,
        0.025382026214404,   0.025382026214404,   0.025382026214404,
        0.025382026214404,   0.025382026214404,   0.025382026214404,
        0.0087509389402756,  0.0087509389402756,  0.0087509389402756,
        0.00066694512451,    0.00066694512451,    0.00066694512451,
        0.00017137575823624, 0.00017137575823624, 0.00017137575823624,
        6.8484179743242e-06, 6.8484179743242e-06, 6.8484179743242e-06,
        20.538835907703,     20.538835907703,     20.538835907703,
        1.2024899663097,     1.2024899663097,     1.2024899663097,
        1.1946931635418,     1.1946931635418,     1.1946931635418,
        0.36938283004206,    0.36938283004206,    0.36938283004206,
        0.028834889384336,   0.028834889384336,   0.028834889384336,
        0.0287633310895,     0.0287633310895,     0.0287633310895,
        0.0287633310895,     0.0287633310895,     0.0287633310895,
        0.0099285992018056,  0.0099285992018056,  0.0099285992018056,
        0.00075519594407854, 0.00075519594407854, 0.00075519594407854,
        0.0001990403696419,  0.0001990403696419,  0.0001990403696419,
        9.8568978271595e-06, 9.8568978271595e-06, 9.8568978271595e-06,
    };
    static double const expected_pstep_mfp[] = {
        0.00013592250371663, 0.00013592250371663, 0.00013592250371663,
        0.37584332248929,    0.37584332248929,    0.37584332248929,
        0.83253192820357,    0.83253192820357,    0.83253192820357,
        0.11286143105278,    0.11286143105278,    0.11286143105278,
        3.0717812835123,     3.0717812835123,     3.0717812835123,
        0.0039397957891655,  0.0039397957891655,  0.0039397957891655,
        3.9397957891655e-06, 3.9397957891655e-06, 3.9397957891655e-06,
        3.6138548768849,     3.6138548768849,     3.6138548768849,
        4.1671931173025,     4.1671931173025,     4.1671931173025,
        4.3305799858826,     4.3305799858826,     4.3305799858826,
        4.5504361595664,     4.5504361595664,     4.5504361595664,
        0.00013592250371663, 0.00013592250371663, 0.00013592250371663,
        0.34290764293478,    0.34290764293478,    0.34290764293478,
        0.77700919805933,    0.77700919805933,    0.77700919805933,
        0.10190349127953,    0.10190349127953,    0.10190349127953,
        2.7320418309273,     2.7320418309273,     2.7320418309273,
        0.0034766487820496,  0.0034766487820496,  0.0034766487820496,
        3.4766487820496e-06, 3.4766487820496e-06, 3.4766487820496e-06,
        3.1528702589869,     3.1528702589869,     3.1528702589869,
        3.444827371237,      3.444827371237,      3.444827371237,
        3.3962551216488,     3.3962551216488,     3.3962551216488,
        2.6426830904199,     2.6426830904199,     2.6426830904199,
    };
    static double const expected_range_mfp[] = {
        0.22220936675071, 0.22220936675071, 0.22220936675071, 0.83037386174787,
        0.83037386174787, 0.83037386174787, 0.83253192820357, 0.83253192820357,
        0.83253192820357, 1.3608709008006,  1.3608709008006,  1.3608709008006,
        3.0717812835123,  3.0717812835123,  3.0717812835123,  3.073482730317,
        3.073482730317,   3.073482730317,   3.073482730317,   3.073482730317,
        3.073482730317,   3.6138548768849,  3.6138548768849,  3.6138548768849,
        4.1671931173025,  4.1671931173025,  4.1671931173025,  4.3305799858826,
        4.3305799858826,  4.3305799858826,  4.5504361595664,  4.5504361595664,
        4.5504361595664,  0.22422024425078, 0.22422024425078, 0.22422024425078,
        0.77626225573414, 0.77626225573414, 0.77626225573414, 0.77700919805933,
        0.77700919805933, 0.77700919805933, 1.267604442164,   1.267604442164,
        1.267604442164,   2.7320418309273,  2.7320418309273,  2.7320418309273,
        2.7333853405003,  2.7333853405003,  2.7333853405003,  2.7333853405003,
        2.7333853405003,  2.7333853405003,  3.1528702589869,  3.1528702589869,
        3.1528702589869,  3.444827371237,   3.444827371237,   3.444827371237,
        3.3962551216488,  3.3962551216488,  3.3962551216488,  2.6426830904199,
        2.6426830904199,  2.6426830904199,
    };
    static double const expected_msc_range_mfp[] = {
        1.7375156528881e-06,
        -1,
        -1,
        3.6924297069194e-05,
        -1,
        -1,
        3.7296268117771e-05,
        -1,
        -1,
        0.0001201548567813,
        -1,
        -1,
        0.0012395873865631,
        0.0012395873865631,
        -1,
        0.0012421038812226,
        0.0012421038812226,
        -1,
        0.0012421038812226,
        0.0012421038812226,
        -1,
        0.0028876979498507,
        0.0028876979498507,
        -1,
        0.016036397835388,
        0.016036397835388,
        -1,
        0.035782778085125,
        0.035782778085125,
        -1,
        0.2261621076249,
        0.2261621076249,
        -1,
        8.1327526181451e-07,
        -1,
        -1,
        1.7283077315793e-05,
        -1,
        -1,
        1.7457185014573e-05,
        -1,
        -1,
        5.6240628649686e-05,
        -1,
        -1,
        0.000580211035609,
        0.000580211035609,
        -1,
        0.00058138892592017,
        0.00058138892592017,
        -1,
        0.00058138892592017,
        0.00058138892592017,
        -1,
        0.0013516386469972,
        0.0013516386469972,
        -1,
        0.007506122679504,
        0.007506122679504,
        -1,
        0.016748768949078,
        0.016748768949078,
        -1,
        0.10585921743233,
        0.10585921743233,
        -1,
    };
    static double const expected_tstep_frac[] = {
        1,
        1,
        1,
        0.42818528370817,
        1,
        1,
        0.13149182096846,
        1,
        1,
        0.81290883531309,
        1,
        1,
        0.030487038712218,
        0.23758713960221,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.038635618762528,
        0.16306418435324,
        1,
        0.04222137550473,
        0.15389074354047,
        1,
        0.03711207636575,
        0.13020806080187,
        1,
        0.049701193400867,
        0.12778216243367,
        1,
        1,
        1,
        1,
        0.41900712918057,
        1,
        1,
        0.1858976449189,
        1,
        1,
        0.82822166142046,
        1,
        1,
        0.036194051101378,
        0.20393678514156,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.041205130690638,
        0.17629883432799,
        1,
        0.043144964461046,
        0.1857764715861,
        1,
        0.048534823151177,
        0.18337917281357,
        1,
        0.040057477120919,
        0.23001362050101,
        1,
    };
    static double const expected_gstep_frac[] = {
        0.99993204182719, 0.99993204182719, 0.99993204182719, 0.9149832761553,
        0.78950804632155, 0.78950804632155, 0.9431472456918,  0.54569308431111,
        0.54569308431111, 0.95390209023009, 0.94323702606103, 0.94323702606103,
        0.95460335692577, 0.68724333161564, 0.24559275913204, 0.99803268655784,
        0.99803268655784, 0.99803268655784, 0.99999803010469, 0.99999803010469,
        0.99999803010469, 0.93332704289517, 0.74451829500646, 0.21673850320042,
        0.91696792030158, 0.72726011451891, 0.19352866775029, 0.92377901839277,
        0.75583209294663, 0.18759684736903, 0.89498283406445, 0.74979249304054,
        0.18016602141734, 0.99993204182719, 0.99993204182719, 0.99993204182719,
        0.92400720850874, 0.80628421156603, 0.80628421156603, 0.92352120903801,
        0.56274328860655, 0.56274328860655, 0.95751458288491, 0.94863624356558,
        0.94863624356558, 0.95214832929963, 0.75033857465424, 0.26794983692654,
        0.99826368837371, 0.99826368837371, 0.99826368837371, 0.99999826167762,
        0.99999826167762, 0.99999826167762, 0.93776670670548, 0.75546138776458,
        0.24079731309592, 0.92923535389169, 0.72526458967012, 0.22498061600122,
        0.92192956951975, 0.73133464761437, 0.22746632584529, 0.94888968703467,
        0.73291272677833, 0.27452292037975,
    };
    static double const expected_alpha_over_mfp[] = {
        0,
        0,
        0,
        1.9277474758756,
        1.7131733429566,
        1.7131733429566,
        1.9720311051209,
        1.2011551342635,
        1.2011551342635,
        1.1554033248142,
        1.1506479671636,
        1.1506479671636,
        0,
        0.38647414398987,
        0.3255440110165,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.27671282717971,
        0.27671282717971,
        0,
        0.23996968027422,
        0.23996968027422,
        0,
        0.23091595196485,
        0.23091595196485,
        0,
        0.21975915383357,
        0.21975915383357,
        0,
        0,
        0,
        2.0653101912806,
        1.8433611195965,
        1.8433611195965,
        2.0784963895716,
        1.2869860517708,
        1.2869860517708,
        1.2333984353282,
        1.2290495735821,
        1.2290495735821,
        0,
        0.42942115017057,
        0.36602660643032,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.31717131307564,
        0.31717131307564,
        0,
        0.29029030840547,
        0.29029030840547,
        0,
        0.29444195567809,
        0.29444195567809,
        0,
        0.37840329914137,
        0.37840329914137,
    };
    static double const expected_angle[] = {
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0.0023, 0.0037, 0.0037, 0.0082, 0.011,  0.0206, 0.0509, 0.1731, 0.7265,
        0.0208, 0.0187, 0.0221, 0.0307, 0.0516, 0.0898, 0.1481, 0.2403, 0.3779,
        0.0175, 0.0182, 0.023,  0.0305, 0.0525, 0.0903, 0.1439, 0.2443, 0.3798,
        0.0009, 0.0016, 0.0015, 0.0034, 0.0068, 0.0107, 0.0261, 0.0964, 0.8526,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.001,  0.0009, 0.0017, 0.003,  0.0041, 0.0064, 0.016,  0.0655, 0.9014,
        0.001,  0.0014, 0.0023, 0.0032, 0.006,  0.0101, 0.0229, 0.0855, 0.8676,
        0.0011, 0.0015, 0.0015, 0.0031, 0.0055, 0.0091, 0.023,  0.0888, 0.8664,
        0.0008, 0.0016, 0.0019, 0.0021, 0.003,  0.0089, 0.0173, 0.0621, 0.9023,
        0.0251, 0.0292, 0.0362, 0.0545, 0.0759, 0.1182, 0.1631, 0.2154, 0.2824,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0.0001, 0.0004, 0,      0.0004, 0.9991,
        0,      0.0001, 0,      0,      0.0001, 0,      0.0002, 0.0007, 0.9989,
        0,      0,      0,      0,      0.0001, 0.0001, 0.0001, 0.0009, 0.9988,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0.0021, 0.0031, 0.0038, 0.0051, 0.0079, 0.0127, 0.0309, 0.0991, 0.8353,
        0.0196, 0.0218, 0.0268, 0.0341, 0.0563, 0.091,  0.1552, 0.2426, 0.3526,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0031, 0.0035, 0.005,  0.0068, 0.01,   0.0183, 0.0393, 0.1157, 0.7983,
        0.0226, 0.0225, 0.0296, 0.0416, 0.0571, 0.0983, 0.1618, 0.2428, 0.3237,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0033, 0.0036, 0.0053, 0.0072, 0.0084, 0.0173, 0.0321, 0.1038, 0.819,
        0.0168, 0.0192, 0.0204, 0.033,  0.05,   0.0915, 0.149,  0.2479, 0.3722,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0067, 0.0071, 0.0091, 0.0119, 0.0154, 0.0234, 0.0426, 0.1209, 0.7629,
        0.0163, 0.0167, 0.0232, 0.0357, 0.0524, 0.0948, 0.1482, 0.2465, 0.3662,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0,      0.0001, 0,      0,      0,      0,      0,      0,      0.9999,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0.0023, 0.0033, 0.004,  0.005,  0.011,  0.0162, 0.0444, 0.1615, 0.7523,
        0.0136, 0.0163, 0.0177, 0.0244, 0.0438, 0.0884, 0.1436, 0.2526, 0.3996,
        0.0143, 0.0144, 0.0166, 0.0249, 0.0433, 0.0779, 0.1457, 0.2601, 0.4028,
        0.0023, 0.0027, 0.0046, 0.0058, 0.0091, 0.0192, 0.042,  0.1585, 0.7558,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0005, 0.0014, 0.0014, 0.002,  0.0042, 0.0056, 0.0156, 0.059,  0.9103,
        0.0013, 0.0014, 0.002,  0.0037, 0.0045, 0.0099, 0.021,  0.0831, 0.8731,
        0.0011, 0.0012, 0.0023, 0.0028, 0.0063, 0.0092, 0.0213, 0.0814, 0.8744,
        0.0013, 0.0017, 0.0018, 0.0026, 0.0046, 0.0093, 0.0168, 0.073,  0.8889,
        0.0194, 0.0181, 0.0224, 0.0327, 0.0535, 0.0942, 0.1521, 0.2403, 0.3673,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0.0001, 0.0001, 0.0001, 0.0002, 0.0007, 0.9988,
        0,      0,      0,      0.0001, 0,      0.0001, 0,      0.0011, 0.9987,
        0,      0,      0.0001, 0,      0.0001, 0.0001, 0.0002, 0.0006, 0.9989,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0,      0,      0,      0,      0,      0,      0,      0,      1,
        0.0017, 0.0024, 0.0038, 0.0037, 0.0077, 0.0154, 0.0304, 0.0949, 0.84,
        0.018,  0.018,  0.0222, 0.0352, 0.0536, 0.0897, 0.1505, 0.2467, 0.3661,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0029, 0.0027, 0.0028, 0.0062, 0.0099, 0.016,  0.0345, 0.1028, 0.8222,
        0.0201, 0.0238, 0.0281, 0.0439, 0.0649, 0.098,  0.1582, 0.2321, 0.3309,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.003,  0.0039, 0.0042, 0.0068, 0.0123, 0.016,  0.0369, 0.1041, 0.8128,
        0.0199, 0.0224, 0.0291, 0.0403, 0.0626, 0.0959, 0.1568, 0.2363, 0.3367,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
        0.0013, 0.0027, 0.0025, 0.0038, 0.005,  0.0102, 0.0207, 0.0548, 0.899,
        0.0221, 0.023,  0.026,  0.0415, 0.0598, 0.0988, 0.1547, 0.2309, 0.3432,
        0,      0,      0,      0,      0,      0,      0,      0,      0,
    };
    static double const expected_displace_frac[] = {
        1.1567584797188e-05,
        0,
        0,
        0.47401939442723,
        0,
        0,
        0.26561408940866,
        0,
        0,
        0.20100287631749,
        0,
        0,
        0.20364304493707,
        0.38701538357109,
        0,
        0.0018031615824536,
        0.00018031615824542,
        0,
        0,
        0,
        0,
        0.36593922352404,
        0.2871903335796,
        0,
        0.51242374432424,
        0.32131587609593,
        0,
        0.44926132990794,
        0.26952098004841,
        0,
        0.73650464453294,
        0.2808597866398,
        0,
        1.1567584797188e-05,
        0,
        0,
        0.40106150935751,
        0,
        0,
        0.4044286412699,
        0,
        0,
        0.17767689446857,
        0,
        0,
        0.22062379940867,
        0.26887011219501,
        0,
        0.0014949405240841,
        0.00014949405240842,
        0,
        0,
        0,
        0,
        0.3293384162944,
        0.26585679389823,
        0,
        0.40088629967887,
        0.32163715148727,
        0,
        0.4661102348845,
        0.3100769740721,
        0,
        0.24389305748632,
        0.30188146067728,
        0,
    };
    static double const expected_action[] = {
        0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
        1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
    };
    static double const expected_avg_engine_samples[] = {
        0, 12,      0, 8,  0, 7.9998,  4, 12, 0, 6,  0, 6,       4, 12,
        0, 0,       0, 0,  4, 12,      0, 8,  0, 8,  4, 12,      4, 10,
        0, 0,       0, 12, 0, 12,      0, 8,  0, 8,  0, 8,       0, 8,
        4, 12,      4, 10, 0, 0,       4, 12, 4, 10, 0, 0,       4, 11.9994,
        4, 10,      0, 0,  0, 11.9536, 4, 10, 0, 0,  0, 11.9998, 0, 8,
        0, 8,       4, 12, 0, 6,       0, 6,  4, 12, 0, 0,       0, 0,
        4, 12,      0, 8,  0, 8,       4, 12, 4, 10, 0, 0,       0, 12,
        0, 12,      0, 8,  0, 8,       0, 8,  0, 8,  4, 12,      4, 10,
        0, 0,       4, 12, 4, 10,      0, 0,  4, 12, 4, 10,      0, 0,
        0, 11.9992, 4, 10, 0, 0,
    };

    EXPECT_VEC_SOFT_EQ(expected_lambda_cm, lambda_cm);
    EXPECT_VEC_SOFT_EQ(expected_pstep_mfp, pstep_mfp);
    EXPECT_VEC_SOFT_EQ(expected_range_mfp, range_mfp);

    EXPECT_VEC_SOFT_EQ(expected_msc_range_mfp, msc_range_mfp);
    EXPECT_VEC_SOFT_EQ(expected_tstep_frac, tstep_frac);
    EXPECT_VEC_SOFT_EQ(expected_gstep_frac, gstep_frac);
    EXPECT_VEC_SOFT_EQ(expected_alpha_over_mfp, alpha_over_mfp);

    EXPECT_VEC_NEAR(expected_angle, angle, 2e-12);
    EXPECT_VEC_NEAR(expected_displace_frac,
                    displace_frac,
                    using_vecgeom_surface ? 1e-2 : 1e-12);
    EXPECT_VEC_EQ(expected_action, action);
    EXPECT_VEC_EQ(expected_avg_engine_samples, avg_engine_samples);
    CELER_DISCARD(using_vecgeom_surface);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
