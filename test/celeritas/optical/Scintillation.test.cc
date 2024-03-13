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
#include "celeritas/Types.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalPrimary.hh"
#include "celeritas/optical/ScintillationData.hh"
#include "celeritas/optical/ScintillationGenerator.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "DiagnosticRngEngine.hh"
#include "OpticalTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ScintillationTest : public OpticalTestBase
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
        ImportScintData spectrum;
        spectrum.material.yield = 5;
        spectrum.material.resolution_scale = 1;
        spectrum.material.components.push_back(
            {0.65713, 128 * nm, 10 * nm, 10 * ns, 6 * ns});
        spectrum.material.components.push_back(
            {0.31987, 128 * nm, 10 * nm, 10 * ns, 1500 * ns});
        spectrum.material.components.push_back(
            {0.023, 200 * nm, 20 * nm, 10 * ns, 3000 * ns});

        ScintillationParams::Input inp;
        inp.matid_to_optmatid.push_back(OpticalMaterialId(0));
        inp.data.push_back(std::move(spectrum));
        params = std::make_shared<ScintillationParams>(
            std::move(inp), this->particle_params());

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

    std::vector<ImportScintComponent> make_material_components()
    {
        std::vector<ImportScintComponent> comps;
        comps.push_back({0.65713, 128 * nm, 10 * nm, 10 * ns, 6 * ns});
        comps.push_back({0.31987, 128 * nm, 10 * nm, 10 * ns, 1500 * ns});
        comps.push_back({0.023, 200 * nm, 20 * nm, 10 * ns, 3000 * ns});
        return comps;
    }

  protected:
    // Host/device storage and reference
    std::shared_ptr<ScintillationParams const> params;

    OpticalMaterialId material{0};
    units::ElementaryCharge charge{-1};

    RandomEngine rng_;
    OpticalDistributionData dist_;

    static constexpr double nm = 1e-9 * units::meter;
    static constexpr double ns = 1e-9 * units::second;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ScintillationTest, params)
{
    auto const& data = params->host_ref();

    EXPECT_EQ(1, data.num_materials);
    EXPECT_EQ(0, data.num_particles);

    auto const& material = data.materials[OpticalMaterialId{0}];
    EXPECT_REAL_EQ(5, material.yield);
    EXPECT_REAL_EQ(1, data.resolution_scale[OpticalMaterialId{0}]);
    EXPECT_EQ(3, data.components.size());

    std::vector<real_type> yield_fracs;
    std::vector<real_type> lambda_means;
    std::vector<real_type> lambda_sigmas;
    std::vector<real_type> rise_times;
    std::vector<real_type> fall_times;
    for (auto idx : material.components)
    {
        auto const& comp = data.components[idx];
        yield_fracs.push_back(comp.yield_frac);
        lambda_means.push_back(comp.lambda_mean);
        lambda_sigmas.push_back(comp.lambda_sigma);
        rise_times.push_back(comp.rise_time);
        fall_times.push_back(comp.fall_time);
    }

    real_type norm{0};
    for (auto const& comp : this->make_material_components())
    {
        norm += comp.yield;
    }
    std::vector<real_type> expected_yield_fracs, expected_lambda_means,
        expected_lambda_sigmas, expected_rise_times, expected_fall_times;
    for (auto const& comp : this->make_material_components())
    {
        expected_yield_fracs.push_back(comp.yield / norm);
        expected_lambda_means.push_back(comp.lambda_mean);
        expected_lambda_sigmas.push_back(comp.lambda_sigma);
        expected_rise_times.push_back(comp.rise_time);
        expected_fall_times.push_back(comp.fall_time);
    }

    EXPECT_VEC_EQ(expected_yield_fracs, yield_fracs);
    EXPECT_VEC_EQ(expected_lambda_means, lambda_means);
    EXPECT_VEC_EQ(expected_lambda_sigmas, lambda_sigmas);
    EXPECT_VEC_EQ(expected_rise_times, rise_times);
    EXPECT_VEC_EQ(expected_fall_times, fall_times);
}

//---------------------------------------------------------------------------//
TEST_F(ScintillationTest, basic)
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
        real_type const expected_energy[] = {9.3561354787881e-06,
                                             9.39574581587642e-06,
                                             1.09240249982534e-05,
                                             6.16620934051192e-06};

        real_type const expected_time[] = {7.30250028666843e-09,
                                           1.05142594015847e-08,
                                           3.11699961936832e-06,
                                           2.68409417173788e-06};

        real_type const expected_cos_theta[] = {0.937735542248463,
                                                -0.775070967887639,
                                                0.744857640134601,
                                                -0.748206733055997};

        real_type const expected_polarization_x[] = {-0.714016941727313,
                                                     0.74609610658139,
                                                     -0.456101107552679,
                                                     0.0275013929040768};

        real_type const expected_cos_polar[] = {0, 0, 0, 0};

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_time, time);
        EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_polarization_x, polarization_x);
        EXPECT_VEC_SOFT_EQ(expected_cos_polar, cos_polar);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ScintillationTest, stress_test)
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

    for (auto i : data.materials[dist_.material].components)
    {
        expected_lambda += data.components[i].lambda_mean
                           * data.components[i].yield_frac;
        expected_error += data.components[i].lambda_sigma
                          * data.components[i].yield_frac;
    }
    EXPECT_SOFT_NEAR(avg_lambda, expected_lambda, expected_error);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
