//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Cherenkov.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/CherenkovDndxCalculator.hh"
#include "celeritas/optical/CherenkovGenerator.hh"
#include "celeritas/optical/CherenkovOffload.hh"
#include "celeritas/optical/CherenkovParams.hh"
#include "celeritas/optical/GeneratorDistributionData.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "OpticalTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
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

using InvCmLength = RealQuantity<InvCentimeter>;
using celeritas::test::from_cm;

//---------------------------------------------------------------------------//
/*!
 * Tabulated refractive index in water as a function of photon wavelength [μm].
 *
 * M. Daimon and A. Masumura. Measurement of the refractive index of distilled
 * water from the near-infrared region to the ultraviolet region, Appl. Opt.
 * 46, 3811-3820 (2007) via refractiveindex.info
 *
 * See G4OpticalMaterialProperties.hh.
 */
Span<real_type const> get_wavelength()
{
    static real_type const wavelength[] = {
        1.129,  1.12,   1.11,   1.101,  1.091,  1.082,  1.072,  1.063,  1.053,
        1.044,  1.034,  1.025,  1.015,  1.006,  0.9964, 0.987,  0.9775, 0.968,
        0.9585, 0.9491, 0.9396, 0.9301, 0.9207, 0.9112, 0.9017, 0.8923, 0.8828,
        0.8733, 0.8638, 0.8544, 0.8449, 0.8354, 0.826,  0.8165, 0.807,  0.7976,
        0.7881, 0.7786, 0.7691, 0.7597, 0.7502, 0.7407, 0.7313, 0.7218, 0.7123,
        0.7029, 0.6934, 0.6839, 0.6744, 0.665,  0.6555, 0.646,  0.6366, 0.6271,
        0.6176, 0.6082, 0.5987, 0.5892, 0.5797, 0.5703, 0.5608, 0.5513, 0.5419,
        0.5324, 0.5229, 0.5135, 0.504,  0.4945, 0.485,  0.4756, 0.4661, 0.4566,
        0.4472, 0.4377, 0.4282, 0.4188, 0.4093, 0.3998, 0.3903, 0.3809, 0.3714,
        0.3619, 0.3525, 0.343,  0.3335, 0.3241, 0.3146, 0.3051, 0.2956, 0.2862,
        0.2767, 0.2672, 0.2578, 0.2483, 0.2388, 0.2294, 0.2199, 0.2104, 0.2009,
        0.1915, 0.182};
    return make_span(wavelength);
}

Span<real_type const> get_refractive_index()
{
    static real_type const refractive_index[]
        = {1.3235601610672, 1.3236962786529, 1.3238469492274, 1.3239820826015,
           1.3241317601229, 1.3242660923031, 1.3244149850321, 1.3245487081924,
           1.3246970353146, 1.3248303521764, 1.3249783454392, 1.3251114708334,
           1.3252593763883, 1.3253925390161, 1.3255346928953, 1.3256740639273,
           1.3258151661284, 1.3259565897464, 1.326098409446,  1.3262392023332,
           1.32638204417,   1.3265255240887, 1.3266682080154, 1.3268132228682,
           1.3269591507928, 1.32710453999,   1.3272525883205, 1.3274018651452,
           1.3275524865531, 1.3277029655807, 1.3278566311755, 1.3280120256415,
           1.328167625867,  1.3283268916356, 1.3284883366632, 1.3286503921034,
           1.3288166823394, 1.3289856845931, 1.3291575989438, 1.3293307783594,
           1.3295091314406, 1.329691073075,  1.3298748828499, 1.3300647424335,
           1.330259008797,  1.3304558735667, 1.3306598562207, 1.3308692454666,
           1.3310844250714, 1.3313034432243, 1.3315313994219, 1.3317664745307,
           1.3320065870964, 1.3322573970809, 1.3325169923974, 1.3327831408348,
           1.3330622051201, 1.3333521716563, 1.3336538750639, 1.3339648469612,
           1.334292688017,  1.3346352438404, 1.3349898436519, 1.3353653263299,
           1.3357594410975, 1.3361692982684, 1.3366053508081, 1.3370652823778,
           1.3375512404603, 1.3380600434506, 1.3386051585073, 1.3391843066628,
           1.3397941348754, 1.34045134693,   1.3411539035636, 1.341898413271,
           1.3427061376724, 1.3435756703017, 1.3445141685829, 1.3455187528254,
           1.3466202523109, 1.3478194943997, 1.3491150472655, 1.350549622307,
           1.3521281492629, 1.3538529543346, 1.3557865447701, 1.3579431129972,
           1.3603615197762, 1.3630595401556, 1.3661548299831, 1.3696980785677,
           1.3737440834249, 1.3785121412586, 1.3841454790718, 1.3908241012126,
           1.399064758142,  1.4093866965284, 1.422764121467,  1.4407913910231,
           1.4679465862259};
    return make_span(refractive_index);
}

// Convert a wavelength in [micrometer] to a photon energy in [MeV]
real_type um_to_mev(real_type wavelength_um)
{
    return value_as<units::MevEnergy>(detail::wavelength_to_energy(
        1e-3 * units::millimeter * wavelength_um));
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CherenkovTest : public ::celeritas::test::OpticalTestBase
{
  protected:
    using Energy = units::MevEnergy;
    using Rng = ::celeritas::test::DiagnosticRngEngine<std::mt19937>;

    void SetUp() override
    {
        // Build optical material: only one material (water)
        ImportOpticalProperty water;
        for (double wl : get_wavelength())
        {
            water.refractive_index.x.push_back(um_to_mev(wl));
        }
        water.refractive_index.y
            = {get_refractive_index().begin(), get_refractive_index().end()};
        water.refractive_index.vector_type = ImportPhysicsVectorType::free;

        MaterialParams::Input input;
        input.properties.push_back(std::move(water));
        input.volume_to_mat = {OpticalMaterialId{0}};
        input.optical_to_core = {CoreMaterialId{0}};
        material = std::make_shared<MaterialParams>(std::move(input));

        // Build Cherenkov data
        params = std::make_shared<CherenkovParams>(*material);
    }

    std::shared_ptr<MaterialParams const> material;
    std::shared_ptr<CherenkovParams const> params;
    OpticalMaterialId material_id{0};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CherenkovTest, angle_integral)
{
    // Check conversion: 1 μm wavelength is approximately 1.2398 eV
    EXPECT_SOFT_EQ(1.2398419843320026e-6, um_to_mev(1));

    auto const& grid = params->host_ref().angle_integral[material_id];
    EXPECT_TRUE(grid);

    auto const& energy = params->host_ref().reals[grid.grid];
    EXPECT_EQ(101, energy.size());
    EXPECT_SOFT_EQ(1.0981771340407463e-6, energy.front());
    EXPECT_SOFT_EQ(6.8123185952307824e-6, energy.back());

    auto const& angle_integral = params->host_ref().reals[grid.value];
    EXPECT_EQ(0, angle_integral.front());
    EXPECT_SOFT_EQ(3.0617629000727e-6, angle_integral.back());
}

//---------------------------------------------------------------------------//

TEST_F(CherenkovTest, dndx)
{
    EXPECT_SOFT_NEAR(369.81e6,
                     constants::alpha_fine_structure * units::Mev::value()
                         * units::centimeter
                         / (constants::hbar_planck * constants::c_light),
                     1e-6);

    MaterialView mat_view{material->host_ref(), material_id};
    CherenkovDndxCalculator calc_dndx(
        mat_view,
        params->host_ref(),
        this->particle_params()->get(ParticleId{0}).charge());

    std::vector<real_type> dndx;
    for (real_type beta :
         {0.5, 0.6813, 0.69, 0.71, 0.73, 0.752, 0.756, 0.8, 0.9, 0.999})
    {
        dndx.push_back(
            native_value_to<InvCmLength>(calc_dndx(units::LightSpeed(beta)))
                .value());
    }
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_dndx[] = {0,
                                               0,
                                               0.57854090574963,
                                               12.39231212654,
                                               41.749688597206,
                                               111.83329546162,
                                               132.04572253875,
                                               343.97410323066,
                                               715.28213549221,
                                               978.60864329219};
        EXPECT_VEC_SOFT_EQ(expected_dndx, dndx);
    }
}

//---------------------------------------------------------------------------//

TEST_F(CherenkovTest, TEST_IF_CELERITAS_DOUBLE(pre_generator))
{
    Rng rng;
    MaterialView const mat_view{material->host_ref(), material_id};

    // 500 keV e-
    {
        // Pre-step values
        OffloadPreStepData pre_step;
        pre_step.pos = {0, 0, 0};
        pre_step.speed = units::LightSpeed{0.63431981443206786};
        pre_step.time = 0;
        pre_step.material = material_id;

        // Post-step values
        auto particle
            = this->make_particle_track_view(Energy{0.5}, pdg::electron());
        auto sim = this->make_sim_track_view(0.15);
        Real3 pos = {sim.step_length(), 0, 0};

        CherenkovOffload pre_generate(
            particle, sim, mat_view, pos, params->host_ref(), pre_step);

        size_type num_samples = 10;
        std::vector<size_type> sampled_num_photons;
        for ([[maybe_unused]] auto i : range(num_samples))
        {
            auto const result = pre_generate(rng);
            CELER_ASSERT(result);
            sampled_num_photons.push_back(result.num_photons);

            // Remaining values are assigned to result from input data
            EXPECT_EQ(pre_step.time, result.time);
            EXPECT_EQ(particle.charge().value(), result.charge.value());
            EXPECT_EQ(material_id, result.material);
            EXPECT_EQ(sim.step_length(), result.step_length);
            EXPECT_EQ(pre_step.speed.value(),
                      result.points[StepPoint::pre].speed.value());
            EXPECT_EQ(particle.speed().value(),
                      result.points[StepPoint::post].speed.value());
            EXPECT_VEC_EQ(pre_step.pos, result.points[StepPoint::pre].pos);
            EXPECT_VEC_EQ(pos, result.points[StepPoint::post].pos);
        }

        // Only number of photons is sampled
        static size_type const expected_num_photons[]
            = {15, 17, 11, 15, 14, 19, 23, 13, 10, 12};
        EXPECT_VEC_EQ(expected_num_photons, sampled_num_photons);
    }

    // Below Cherenkov threshold
    {
        // Pre-step values
        OffloadPreStepData pre_step;
        pre_step.pos = {0, 0, 0};
        pre_step.speed = units::LightSpeed{0.55};
        pre_step.time = 0;
        pre_step.material = material_id;

        // Post-step values
        auto particle
            = this->make_particle_track_view(Energy{0.1}, pdg::electron());
        auto sim = this->make_sim_track_view(0.1);
        Real3 pos = {sim.step_length(), 0, 0};

        CherenkovOffload pre_generate(
            particle, sim, mat_view, pos, params->host_ref(), pre_step);
        auto const result = pre_generate(rng);

        EXPECT_FALSE(result);
        EXPECT_EQ(0, result.num_photons);
    }
}

//---------------------------------------------------------------------------//

TEST_F(CherenkovTest, TEST_IF_CELERITAS_DOUBLE(generator))
{
    Rng rng;
    MaterialView mat_view{material->host_ref(), material_id};

    // Mean values
    real_type avg_costheta;
    real_type avg_energy;
    real_type avg_displacement;
    real_type avg_engine_samples;  // !< per photon!
    real_type total_num_photons;

    // Distributions
    int num_bins = 16;
    std::vector<real_type> costheta_dist(num_bins);
    std::vector<real_type> energy_dist(num_bins);
    std::vector<real_type> displacement_dist(num_bins);

    // Energy distribution binning
    real_type emin = um_to_mev(get_wavelength().front());
    real_type emax = um_to_mev(get_wavelength().back());
    real_type edel = (emax - emin) / num_bins;

    auto sample = [&](OffloadPreStepData& pre_step,
                      ParticleTrackView const& particle,
                      SimTrackView const& sim,
                      Real3 const& pos,
                      size_type num_samples) {
        // Reset tallies
        rng.reset_count();
        avg_costheta = avg_energy = avg_displacement = total_num_photons = 0;
        std::fill(costheta_dist.begin(), costheta_dist.end(), 0);
        std::fill(energy_dist.begin(), energy_dist.end(), 0);
        std::fill(displacement_dist.begin(), displacement_dist.end(), 0);

        // Displacement distribution binning
        real_type dmin = 0;
        real_type dmax = sim.step_length();
        real_type ddel = (dmax - dmin) / num_bins;

        // Calculate the average number of photons produced per unit length
        CherenkovOffload pre_generate(
            particle, sim, mat_view, pos, params->host_ref(), pre_step);

        Real3 inc_dir = make_unit_vector(pos - pre_step.pos);
        for (size_type i = 0; i < num_samples; ++i)
        {
            auto const dist = pre_generate(rng);
            CELER_ASSERT(dist);

            // Sample the optical photons
            CherenkovGenerator generate_photon(
                mat_view, params->host_ref(), dist);

            for (size_type j = 0; j < dist.num_photons; ++j)
            {
                auto photon = generate_photon(rng);
                // Bin cos(theta) of the photon relative to the incident
                // particle direction
                {
                    real_type costheta = dot_product(inc_dir, photon.direction);
                    avg_costheta += costheta;
                    // Remap from [-1,1] to [0,1]
                    int bin = static_cast<int>((1 + costheta) / 2 * num_bins);
                    CELER_ASSERT(bin >= 0 && bin < num_bins);
                    ++costheta_dist[bin];
                }
                // Bin photon energy
                {
                    real_type energy = photon.energy.value();
                    avg_energy += energy;
                    int bin = static_cast<int>((energy - emin) / edel);
                    CELER_ASSERT(bin >= 0 && bin < num_bins);
                    ++energy_dist[bin];
                }
                // Bin photon displacement
                {
                    real_type displacement
                        = distance(pre_step.pos, photon.position);
                    avg_displacement += displacement;
                    int bin = static_cast<int>((displacement - dmin) / ddel);
                    CELER_ASSERT(bin >= 0 && bin < num_bins);
                    ++displacement_dist[bin];
                }

                // Photon polarization is perpendicular to the cone angle
                EXPECT_SOFT_EQ(
                    0, dot_product(photon.direction, photon.polarization));
            }
            total_num_photons += dist.num_photons;
        }
        avg_costheta /= total_num_photons;
        avg_energy /= total_num_photons;
        avg_displacement /= (from_cm(1) * total_num_photons);
        avg_engine_samples = real_type(rng.count()) / total_num_photons;
    };

    size_type num_samples = 64;

    // Photons are emitted on the surface of a cone, with the cone angle
    // measured with respect to the incident particle direction. As the
    // incident energy decreases, the cone angle and the number of photons
    // produced decreases, and the energy of the emitted photons increases.

    // 10 GeV e-
    {
        // Pre-step values
        OffloadPreStepData pre_step;
        pre_step.pos = {0, 0, 0};
        pre_step.speed = units::LightSpeed{0.99999999869453382};  // 10 GeV
        pre_step.time = 0;
        pre_step.material = material_id;

        // Post-step values
        auto particle
            = this->make_particle_track_view(Energy(9999), pdg::electron());
        auto sim = this->make_sim_track_view(1);
        Real3 pos = {sim.step_length(), 0, 0};

        // clang-format off
        static double const expected_costheta_dist[]
            = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52451, 10508, 0};
        static double const expected_energy_dist[]
            = {3690, 3774, 3698, 3752, 3684, 3658, 3768, 3831,
               3921, 4029, 4025, 3941, 4134, 4286, 4307, 4461};
        static double const expected_displacement_dist[]
            = {3909, 4064, 3802, 3920, 4001, 3904, 3891, 3955,
               3999, 3924, 3903, 3900, 3959, 3932, 4023, 3873};
        // clang-format on

        sample(pre_step, particle, sim, pos, num_samples);

        EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
        EXPECT_VEC_EQ(expected_energy_dist, energy_dist);
        EXPECT_VEC_EQ(expected_displacement_dist, displacement_dist);
        EXPECT_SOFT_EQ(0.73055857883146702, avg_costheta);
        EXPECT_SOFT_EQ(4.0497726102182314e-06, avg_energy);
        EXPECT_SOFT_EQ(0.50020101984474064, avg_displacement);
        EXPECT_SOFT_EQ(983.734375, total_num_photons / num_samples);
        EXPECT_SOFT_EQ(10.609603075017075, avg_engine_samples);
    }

    // 500 keV e-: 1/beta_avg ~ 1.336
    {
        // Pre-step values
        OffloadPreStepData pre_step;
        pre_step.pos = {0, 0, 0};
        pre_step.speed = units::LightSpeed(0.86286196322132458);  // 500 keV
        pre_step.time = 0;
        pre_step.material = material_id;

        // Post-step values (150 keV)
        auto particle
            = this->make_particle_track_view(Energy(0.15), pdg::electron());
        EXPECT_SOFT_EQ(0.63431981443206786,
                       value_as<units::LightSpeed>(particle.speed()));
        auto sim = this->make_sim_track_view(0.15);
        Real3 pos = {sim.step_length(), 0, 0};

        static double const expected_costheta_dist[]
            = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 991};
        static double const expected_energy_dist[]
            = {0, 0, 0, 0, 4, 14, 29, 26, 48, 51, 77, 103, 129, 132, 174, 204};
        static double const expected_displacement_dist[] = {
            123, 114, 103, 102, 83, 81, 80, 57, 60, 59, 31, 29, 36, 14, 16, 3};

        sample(pre_step, particle, sim, pos, num_samples);

        EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
        EXPECT_VEC_EQ(expected_energy_dist, energy_dist);
        EXPECT_VEC_EQ(expected_displacement_dist, displacement_dist);
        EXPECT_SOFT_EQ(0.95045221539598979, avg_costheta);
        EXPECT_SOFT_EQ(5.5902203966702514e-06, avg_energy);
        EXPECT_SOFT_EQ(0.049715603846029896, avg_displacement);
        EXPECT_SOFT_EQ(15.484375, total_num_photons / num_samples);
        EXPECT_SOFT_EQ(25.077699293642784, avg_engine_samples);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
