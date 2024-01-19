//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationGenerator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalPrimary.hh"
#include "celeritas/optical/ScintillationData.hh"
#include "celeritas/optical/ScintillationGenerator.hh"
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
        // Test scintillation spectrum: only one (LAr) material
        HostVal<ScintillationData> data;
        ScintillationSpectrum spectrum;
        static constexpr double nm = 1e-9 * units::meter;
        static constexpr double ns = 1e-9 * units::second;

        spectrum.yield_prob = {0.75, 0.25, 0};
        spectrum.lambda_mean = {128 * nm, 128 * nm, 0};
        spectrum.lambda_sigma = {10 * nm, 10 * nm, 0};
        spectrum.rise_time = {10 * ns, 10 * ns, 0};
        spectrum.fall_time = {6 * ns, 1500 * ns, 0};

        make_builder(&data.spectrum).push_back(spectrum);
        params = std::make_shared<ScintillationParams>(make_const_ref(data));

        // Test step input
        dist_.num_photons = 4;
        dist_.step_length = 1 * units::centimeter;
        dist_.time = 0;
        dist_.points[StepPoint::pre].speed = units::LightSpeed(0.99);
        dist_.points[StepPoint::post].speed = units::LightSpeed(0.99 * 0.9);
        dist_.points[StepPoint::pre].pos = {0, 0, 0};
        dist_.points[StepPoint::post].pos = {0, 0, 1};
        dist_.charge = units::ElementaryCharge{-1};
        dist_.material = OpticalMaterialId{0};
    }
    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

  protected:
    // Host/device storage and reference
    std::shared_ptr<ScintillationParams const> params;
    OpticalMaterialId material{0};
    units::ElementaryCharge charge{-1};

    RandomEngine rng_;
    OpticalDistributionData dist_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ScintillationGeneratorTest, basic)
{
    // Output data
    std::vector<OpticalPrimary> storage(dist_.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(
        dist_, params->host_ref(), make_span(storage));
    RandomEngine& rng_engine = this->rng();

    // Generate optical photons for a given input
    auto photons = generate_photons(rng_engine);

    // Check results
    std::vector<real_type> energy;
    std::vector<real_type> time;
    std::vector<real_type> cos_theta;

    for (auto i : range(dist_.num_photons))
    {
        energy.push_back(photons[i].energy.value());
        time.push_back(photons[i].time);
        cos_theta.push_back(
            dot_product(photons[i].direction,
                        dist_.points[StepPoint::post].pos
                            - dist_.points[StepPoint::pre].pos));
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
    dist_.num_photons = 1e+6;

    // Output data
    std::vector<OpticalPrimary> storage(dist_.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(
        dist_, params->host_ref(), make_span(storage));

    // Generate optical photons for a given input
    auto photons = generate_photons(this->rng());

    // Check results
    double avg_lambda{0};
    double hc = ScintillationGenerator::hc() / units::Mev::value();
    for (auto i : range(dist_.num_photons))
    {
        avg_lambda += hc / photons[i].energy.value();
    }
    avg_lambda /= static_cast<double>(dist_.num_photons);

    double expected_lambda{0};
    double expected_error{0};
    ScintillationSpectrum spectrum
        = params->host_ref().spectrum[dist_.material];

    for (auto i : range(3))
    {
        expected_lambda += spectrum.lambda_mean[i] * spectrum.yield_prob[i];
        expected_error += spectrum.lambda_sigma[i] * spectrum.yield_prob[i];
    }
    EXPECT_SOFT_NEAR(avg_lambda, expected_lambda, expected_error);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
