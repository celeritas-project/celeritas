//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationGenerator.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/ScintillationGenerator.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalPrimary.hh"
#include "celeritas/optical/ScintillationData.hh"
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
        // Test scintillation spectrum: only one material with three components
        HostVal<ScintillationData> data;
        static constexpr size_type num_components = 3;
        static constexpr double nm = 1e-9 * units::meter;
        static constexpr double ns = 1e-9 * units::second;

        using Real3 = Array<real_type, num_components>;
        Real3 yield_prob = {0.65713, 0.31987, 1 - 0.65713 - 0.31987};
        Real3 lambda_mean = {128 * nm, 128 * nm, 200 * nm};
        Real3 lambda_sigma = {10 * nm, 10 * nm, 20 * nm};
        Real3 rise_time = {10 * ns, 10 * ns, 10 * ns};
        Real3 fall_time = {6 * ns, 1500 * ns, 3000 * ns};

        std::vector<ScintillationComponent> components;
        for (size_type i = 0; i < num_components; ++i)
        {
            if (yield_prob[i] > 0)
            {
                ScintillationComponent component;
                component.yield_prob = yield_prob[i];
                component.lambda_mean = lambda_mean[i];
                component.lambda_sigma = lambda_sigma[i];
                component.rise_time = rise_time[i];
                component.fall_time = fall_time[i];
                components.push_back(component);
            }
        }

        ScintillationParams::ScintillationInput input;
        input.data.push_back(components);
        params = std::make_shared<ScintillationParams>(input);

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
    std::vector<real_type> polarization_x;
    std::vector<real_type> cos_polar;

    for (auto i : range(dist_.num_photons))
    {
        energy.push_back(photons[i].energy.value());
        time.push_back(photons[i].time / units::second);
        cos_theta.push_back(
            dot_product(photons[i].direction,
                        dist_.points[StepPoint::post].pos
                            - dist_.points[StepPoint::pre].pos));
        polarization_x.push_back(photons[i].polarization[0]);
        cos_polar.push_back(
            dot_product(photons[i].polarization, photons[i].direction));
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        const real_type expected_energy[] = {9.3561354787881e-06,
                                             9.39574581587642e-06,
                                             1.09240249982534e-05,
                                             6.16620934051192e-06};

        const real_type expected_time[] = {7.30250028666843e-09,
                                           1.05142594015847e-08,
                                           3.11699961936832e-06,
                                           2.68409417173788e-06};

        const real_type expected_cos_theta[] = {0.937735542248463,
                                                -0.775070967887639,
                                                0.744857640134601,
                                                -0.748206733055997};

        const real_type expected_polarization_x[] = {-0.714016941727313,
                                                     0.74609610658139,
                                                     -0.456101107552679,
                                                     0.0275013929040768};

        const real_type expected_cos_polar[] = {0, 0, 0, 0};

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_time, time);
        EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_polarization_x, polarization_x);
        EXPECT_VEC_SOFT_EQ(expected_cos_polar, cos_polar);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ScintillationGeneratorTest, stress_test)
{
    // Generate a large number of optical photons
    dist_.num_photons = 123456;

    // Output data
    std::vector<OpticalPrimary> storage(dist_.num_photons);

    // Create the generator
    HostCRef<ScintillationData> data = params->host_ref();
    ScintillationGenerator generate_photons(dist_, data, make_span(storage));

    // Generate optical photons for a given input
    auto photons = generate_photons(this->rng());

    // Check results
    double avg_lambda{0};
    double hc = constants::h_planck * constants::c_light / units::Mev::value();
    for (auto i : range(dist_.num_photons))
    {
        avg_lambda += hc / photons[i].energy.value();
    }
    avg_lambda /= static_cast<double>(dist_.num_photons);

    double expected_lambda{0};
    double expected_error{0};

    for (auto i : data.spectra[dist_.material].components)
    {
        expected_lambda += data.components[i].lambda_mean
                           * data.components[i].yield_prob;
        expected_error += data.components[i].lambda_sigma
                          * data.components[i].yield_prob;
    }
    EXPECT_SOFT_NEAR(avg_lambda, expected_lambda, expected_error);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
