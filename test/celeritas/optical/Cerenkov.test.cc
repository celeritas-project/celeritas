//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Cerenkov.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/grid/VectorUtils.hh"
#include "celeritas/optical/CerenkovDndxCalculator.hh"
#include "celeritas/optical/CerenkovGenerator.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/random/distribution/PoissonDistribution.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
struct InvCentimeter
{
    static CELER_CONSTEXPR_FUNCTION real_type value()
    {
        return 1 / units::centimeter;
    }
    static char const* label() { return "1/cm"; }
};

using InvCmDnDx = Quantity<InvCentimeter>;
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
Span<double const> get_wavelength()
{
    static Array<double, 101> const wavelength = {
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

Span<double const> get_refractive_index()
{
    static Array<double, 101> const refractive_index
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

double convert_to_energy(double wavelength)
{
    return constants::h_planck * constants::c_light / units::Mev::value()
           / wavelength;
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CerenkovTest : public Test
{
  protected:
    void SetUp() override
    {
        this->build_optical_properties();
        properties = make_const_ref(data);
        params = std::make_shared<CerenkovParams>(properties);
    }

    void build_optical_properties();

    static constexpr double micrometer = 1e-4 * units::centimeter;

    HostVal<OpticalPropertyData> data;
    HostCRef<OpticalPropertyData> properties;
    std::shared_ptr<CerenkovParams const> params;
    OpticalMaterialId material{0};
    units::ElementaryCharge charge{1};
};

//---------------------------------------------------------------------------//

void CerenkovTest::build_optical_properties()
{
    auto wavelength = get_wavelength();
    std::vector<double> energy(wavelength.size());
    for (auto i : range(energy.size()))
    {
        energy[i] = convert_to_energy(wavelength[i] * micrometer);
    }
    auto rindex = get_refractive_index();
    CELER_ASSERT(energy.size() == rindex.size());

    // In a dispersive medium the index of refraction is an increasing
    // function of photon energy
    CELER_ASSERT(is_monotonic_increasing(rindex));

    // Only one material: water
    GenericGridData grid;
    auto reals = make_builder(&data.reals);
    grid.grid = reals.insert_back(energy.begin(), energy.end());
    grid.value = reals.insert_back(rindex.begin(), rindex.end());
    make_builder(&data.refractive_index).push_back(grid);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CerenkovTest, angle_integral)
{
    // Check conversion: 1 μm wavelength is approximately 1.2398 eV
    EXPECT_SOFT_EQ(1.2398419843320026e-6, convert_to_energy(1 * micrometer));

    auto const& grid = params->host_ref().angle_integral[material];
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

TEST_F(CerenkovTest, dndx)
{
    EXPECT_SOFT_NEAR(369.81e6,
                     constants::alpha_fine_structure * units::Mev::value()
                         * units::centimeter
                         / (constants::hbar_planck * constants::c_light),
                     1e-6);

    std::vector<real_type> dndx;
    CerenkovDndxCalculator calc_dndx(
        properties, params->host_ref(), material, charge);
    for (real_type beta :
         {0.5, 0.6813, 0.69, 0.71, 0.73, 0.752, 0.756, 0.8, 0.9, 0.999})
    {
        dndx.push_back(
            native_value_to<InvCmDnDx>(calc_dndx(units::LightSpeed(beta)))
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

TEST_F(CerenkovTest, TEST_IF_CELERITAS_DOUBLE(generator))
{
    DiagnosticRngEngine<std::mt19937> rng;

    // Mean valuese
    real_type avg_costheta;
    real_type avg_energy;
    real_type avg_displacement;
    real_type avg_engine_samples;
    real_type total_num_photons;

    // Distributions
    int num_bins = 16;
    std::vector<real_type> costheta_dist(num_bins);
    std::vector<real_type> energy_dist(num_bins);
    std::vector<real_type> displacement_dist(num_bins);

    // Energy distribution binning
    real_type emin = convert_to_energy(get_wavelength().front() * micrometer);
    real_type emax = convert_to_energy(get_wavelength().back() * micrometer);
    real_type edel = (emax - emin) / num_bins;

    auto sample = [&](OpticalDistributionData& dist, size_type num_samples) {
        // Reset tallies
        rng.reset_count();
        avg_costheta = avg_energy = avg_displacement = total_num_photons = 0;
        std::fill(costheta_dist.begin(), costheta_dist.end(), 0);
        std::fill(energy_dist.begin(), energy_dist.end(), 0);
        std::fill(displacement_dist.begin(), displacement_dist.end(), 0);

        // Displacement distribution binning
        real_type dmin = 0;
        real_type dmax = dist.step_length;
        real_type ddel = (dmax - dmin) / num_bins;

        // Calculate the average number of photons produced per unit length
        auto const& pre_step = dist.points[StepPoint::pre];
        auto const& post_step = dist.points[StepPoint::post];
        units::LightSpeed beta(
            0.5 * (pre_step.speed.value() + post_step.speed.value()));
        CerenkovDndxCalculator calc_dndx(
            properties, params->host_ref(), dist.material, dist.charge);
        real_type mean_num_photons = calc_dndx(beta) * dist.step_length;
        CELER_ASSERT(mean_num_photons > 0);

        Real3 inc_dir = make_unit_vector(post_step.pos - pre_step.pos);
        for (size_type i = 0; i < num_samples; ++i)
        {
            // Sample the number of photons from a Poisson distribution
            dist.num_photons = PoissonDistribution(mean_num_photons)(rng);

            // Sample the optical photons
            std::vector<OpticalPrimary> storage(dist.num_photons);
            CerenkovGenerator generate_photons(
                properties, params->host_ref(), dist, make_span(storage));
            auto photons = generate_photons(rng);

            for (auto const& photon : photons)
            {
                // Bin cos(theta) of the photon relative to the incident
                // particle direction
                {
                    real_type costheta = dot_product(inc_dir, photon.direction);
                    avg_costheta += costheta;
                    // Remap from [-1,1] to [0,1]
                    int bin = (1 + costheta) / 2 * num_bins;
                    CELER_ASSERT(bin < num_bins);
                    ++costheta_dist[bin];
                }
                // Bin photon energy
                {
                    real_type energy = photon.energy.value();
                    avg_energy += energy;
                    int bin = (energy - emin) / edel;
                    CELER_ASSERT(bin < num_bins);
                    ++energy_dist[bin];
                }
                // Bin photon displacement
                {
                    real_type displacement
                        = distance(pre_step.pos, photon.position);
                    avg_displacement += displacement;
                    int bin = (displacement - dmin) / ddel;
                    CELER_ASSERT(bin < num_bins);
                    ++displacement_dist[bin];
                }

                // Photon polarization is perpendicular to the cone angle
                EXPECT_SOFT_EQ(
                    0, dot_product(photon.direction, photon.polarization));
            }
            total_num_photons += photons.size();
        }
        avg_costheta /= total_num_photons;
        avg_energy /= total_num_photons;
        avg_displacement /= (units::centimeter * total_num_photons);
        avg_engine_samples = real_type(rng.count()) / num_samples;
    };

    size_type num_samples = 64;

    OpticalDistributionData dist;
    dist.time = 0;
    dist.charge = charge;
    dist.material = material;
    dist.points[StepPoint::pre].pos = {0, 0, 0};

    // Photons are emitted on the surface of a cone, with the cone angle
    // measured with respect to the incident particle direction. As the
    // incident energy decreases, the cone angle and the number of photons
    // produced decreases, and the energy of the emitted photons increases.

    // 10 GeV e-
    {
        dist.points[StepPoint::pre].speed
            = units::LightSpeed(0.99999999869453382);
        dist.points[StepPoint::post].speed
            = units::LightSpeed(0.9999999986942727);
        dist.step_length = 1 * units::centimeter;
        dist.points[StepPoint::post].pos = {dist.step_length, 0, 0};

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

        sample(dist, num_samples);

        EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
        EXPECT_VEC_EQ(expected_energy_dist, energy_dist);
        EXPECT_VEC_EQ(expected_displacement_dist, displacement_dist);
        EXPECT_SOFT_EQ(0.73055857883146702, avg_costheta);
        EXPECT_SOFT_EQ(4.0497726102182314e-06, avg_energy);
        EXPECT_SOFT_EQ(0.50020101984474064, avg_displacement);
        EXPECT_SOFT_EQ(983.734375, total_num_photons / num_samples);
        EXPECT_SOFT_EQ(10437.03125, avg_engine_samples);
    }
    // 500 keV e-
    {
        dist.points[StepPoint::pre].speed
            = units::LightSpeed(0.86286196322132458);
        dist.points[StepPoint::post].speed
            = units::LightSpeed(0.63431981443206786);
        dist.step_length = 0.15 * units::centimeter;
        dist.points[StepPoint::post].pos = {dist.step_length, 0, 0};

        static double const expected_costheta_dist[]
            = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 946};
        static double const expected_energy_dist[]
            = {0, 0, 0, 0, 10, 13, 24, 29, 47, 54, 81, 85, 120, 119, 176, 188};
        static double const expected_displacement_dist[] = {
            108, 108, 90, 105, 83, 88, 85, 65, 49, 43, 31, 29, 31, 16, 13, 2};

        sample(dist, num_samples);

        EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
        EXPECT_VEC_EQ(expected_energy_dist, energy_dist);
        EXPECT_VEC_EQ(expected_displacement_dist, displacement_dist);
        EXPECT_SOFT_EQ(0.95069574770853793, avg_costheta);
        EXPECT_SOFT_EQ(5.5675610907221099e-06, avg_energy);
        EXPECT_SOFT_EQ(0.049432369852608751, avg_displacement);
        EXPECT_SOFT_EQ(14.78125, total_num_photons / num_samples);
        EXPECT_SOFT_EQ(401.5, avg_engine_samples);
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
