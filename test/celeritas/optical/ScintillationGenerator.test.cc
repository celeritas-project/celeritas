//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationGenerator.test.cc
//---------------------------------------------------------------------------//
#include <numeric>

#include "celeritas/optical/ScintillationGenerator.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/OpticalPrimary.hh"
#include "celeritas/optical/ScintillationData.hh"
#include "celeritas/optical/ScintillationInput.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ScintillationGeneratorTest : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using HostValue = HostVal<ScintillationData>;
    //!@}

  protected:
    void SetUp() override
    {
        // Test scintillation spectra: only one (LAr) material
        HostVal<OpticalPropertyData> data;
        ScintillationSpectra spectra;
        static constexpr double nm = 1e-9 * units::meter;
        static constexpr double ns = 1e-9 * units::second;

        spectra.yield_prob = {0.75, 0.25, 0};
        spectra.lambda_mean = {128 * nm, 128 * nm, 0};
        spectra.lambda_sigma = {10 * nm, 10 * nm, 0};
        spectra.rise_time = {10 * ns, 10 * ns, 0};
        spectra.fall_time = {6 * ns, 1500 * ns, 0};

        make_builder(&data.scint_spectra).push_back(spectra);
        properties = make_const_ref(data);
        params = std::make_shared<ScintillationParams>(properties);

        // Test step input
        input_.num_photons = 4;
        input_.step_length = 1 * units::centimeter;
        input_.time = 0;
        input_.pre_velocity = 0.99 * constants::c_light;
        input_.post_velocity = 0.9 * input_.pre_velocity;
        input_.pre_pos = {0, 0, 0};
        input_.post_pos = {0, 0, 1};
        input_.charge = units::ElementaryCharge{-1};
        input_.matId = OpticalMaterialId{0};
    }
    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

  protected:
    // Host/device storage and reference
    HostCRef<OpticalPropertyData> properties;
    std::shared_ptr<ScintillationParams const> params;
    OpticalMaterialId material{0};
    units::ElementaryCharge charge{-1};

    RandomEngine rng_;
    ScintillationInput input_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ScintillationGeneratorTest, basic)
{
    // Output data
    std::vector<OpticalPrimary> storage;
    storage.reserve(input_.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(
        input_, params->host_ref(), make_span(storage));
    RandomEngine& rng_engine = this->rng();

    // Generate optical photons for a given input
    auto photons = generate_photons(rng_engine);

    // Check results
    std::vector<real_type> energy;
    std::vector<real_type> time;
    std::vector<real_type> cos_theta;

    for (auto i : range(input_.num_photons))
    {
        energy.push_back(photons[i].energy.value());
        time.push_back(photons[i].time);
        cos_theta.push_back(dot_product(photons[i].direction,
                                        input_.post_pos - input_.pre_pos));
    }

    const real_type expected_energy[] = {9.35613547878811e-06,
                                         9.39574581587642e-06,
                                         1.09240249982534e-05,
                                         9.10710182050691e-06};

    const real_type expected_time[] = {7.30250028666843e-09,
                                       1.05142594015847e-08,
                                       1.24626439432502e-08,
                                       2.11861667895083e-09};

    const real_type expected_cos_theta[] = {0.937735542248463,
                                            -0.77507096788764,
                                            0.744857640134601,
                                            -0.182537678076479};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_time, time);
    EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
}

//---------------------------------------------------------------------------//

TEST_F(ScintillationGeneratorTest, stress_test)
{
    // Generate a large number of optical photons
    input_.num_photons = 1e+6;

    // Output data
    std::vector<OpticalPrimary> storage;
    storage.reserve(input_.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(
        input_, params->host_ref(), make_span(storage));

    // Generate optical photons for a given input
    auto photons = generate_photons(this->rng());

    // Check results
    std::vector<double> wavelength;
    double energy_to_wavelength = ScintillationGenerator::hc();
    for (auto i : range(input_.num_photons))
    {
        wavelength.push_back(energy_to_wavelength / photons[i].energy.value());
    }
    double avg_wavelength = static_cast<double>(std::reduce(wavelength.begin(),
                                                            wavelength.end()))
                            / wavelength.size();

    double expected_wavelength{0};
    double expected_error{0};
    ScintillationSpectra spectra = params->host_ref().spectra[input_.matId];

    for (auto i : range(3))
    {
        expected_wavelength += spectra.lambda_mean[i] * spectra.yield_prob[i];
        expected_error += spectra.lambda_sigma[i] * spectra.yield_prob[i];
    }
    EXPECT_SOFT_NEAR(avg_wavelength, expected_wavelength, expected_error);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
