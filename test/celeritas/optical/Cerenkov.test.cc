//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Cerenkov.test.cc
//---------------------------------------------------------------------------//
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/optical/CerenkovDndxCalculator.hh"
#include "celeritas/optical/CerenkovGenerator.hh"
#include "celeritas/optical/CerenkovParams.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
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
std::vector<double> const& get_wavelength()
{
    static std::vector<double> const wavelength = {
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
    return wavelength;
}

std::vector<double> const& get_refractive_index()
{
    static std::vector<double> const refractive_index
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
    return refractive_index;
}

double convert_to_energy(double wavelength)
{
    // TODO: 1e-14
    return constants::h_planck * constants::c_light / constants::e_electron
           / wavelength * 1e-14;
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
    real_type charge{1};
    DiagnosticRngEngine<std::mt19937> rng;
};

//---------------------------------------------------------------------------//

void CerenkovTest::build_optical_properties()
{
    auto const& wavelength = get_wavelength();
    std::vector<double> energy(wavelength.size());
    for (auto i : range(energy.size()))
    {
        energy[i] = convert_to_energy(wavelength[i] * micrometer);
    }
    auto const& rindex = get_refractive_index();
    CELER_ASSERT(energy.size() == rindex.size());

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
    EXPECT_EQ(1.2398419843320026e-6, convert_to_energy(1 * micrometer));

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
    EXPECT_SOFT_EQ(369.810177620644e6,
                   constants::alpha_fine_structure * constants::e_electron
                       / (constants::hbar_planck * constants::c_light) * 1e14);

    std::vector<real_type> dndx;
    CerenkovDndxCalculator calc_dndx(
        properties, params->host_ref(), material, charge);
    for (real_type beta :
         {0.5, 0.6813, 0.69, 0.71, 0.73, 0.75, 0.8, 0.9, 0.999})
    {
        dndx.push_back(calc_dndx(1 / beta));
    }

    static double const expected_dndx[] = {0,
                                           0,
                                           0.57854090574963,
                                           12.39231212654,
                                           41.749688597206,
                                           102.71747988865,
                                           343.97410323066,
                                           715.28213549221,
                                           978.60864329219};
    EXPECT_VEC_SOFT_EQ(expected_dndx, dndx);
}

//---------------------------------------------------------------------------//

TEST_F(CerenkovTest, generator)
{
    EXPECT_EQ(0, rng.count());
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
