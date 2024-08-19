//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Scintillation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/GeneratorDistributionData.hh"
#include "celeritas/optical/Primary.hh"
#include "celeritas/optical/ScintillationData.hh"
#include "celeritas/optical/ScintillationGenerator.hh"
#include "celeritas/optical/ScintillationOffload.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "DiagnosticRngEngine.hh"
#include "OpticalTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//

using celeritas::test::from_cm;
using celeritas::test::to_cm;
using TimeSecond = celeritas::Quantity<celeritas::units::Second>;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ScintillationTestBase : public ::celeritas::test::OpticalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using Rng = ::celeritas::test::DiagnosticRngEngine<std::mt19937>;
    using HostValue = HostVal<ScintillationData>;
    using MevEnergy = units::MevEnergy;
    using LightSpeed = units::LightSpeed;
    using SPParams = std::shared_ptr<ScintillationParams>;
    using VecScintComponents = std::vector<ImportScintComponent>;
    //!@}

  protected:
    virtual SPParams build_scintillation_params() = 0;

    //! Set up mock pre-generator step data
    OffloadPreStepData build_pre_step()
    {
        OffloadPreStepData pre_step;
        pre_step.speed = LightSpeed(0.99862874144970537);  // 10 MeV
        pre_step.pos = {0, 0, 0};
        pre_step.time = 0;
        pre_step.material = opt_mat_;
        return pre_step;
    }

  protected:
    OpticalMaterialId opt_mat_{0};

    // Post-step values
    Real3 post_pos_{0, 0, from_cm(1)};
    MevEnergy post_energy_{9.25};
    MevEnergy edep_{0.75};
    real_type step_length_{2.5};  // [cm]
};

class MaterialScintillationTest : public ScintillationTestBase
{
  public:
    //! Create scintillation params
    SPParams build_scintillation_params() override
    {
        ScintillationParams::Input inp;
        inp.resolution_scale.push_back(1);

        // One material, three components
        ImportMaterialScintSpectrum mat_spec;
        mat_spec.yield_per_energy = 5;
        mat_spec.components = this->build_material_components();
        inp.materials.push_back(std::move(mat_spec));

        return std::make_shared<ScintillationParams>(std::move(inp));
    }

    //! Create material components
    std::vector<ImportScintComponent> build_material_components()
    {
        static constexpr real_type nm = units::meter * 1e-9;
        static constexpr real_type ns = units::nanosecond;

        // Note second component has zero rise time
        std::vector<ImportScintComponent> comps;
        comps.push_back({0.5, 100 * nm, 5 * nm, 10 * ns, 6 * ns});
        comps.push_back({0.3, 200 * nm, 10 * nm, 0, 1500 * ns});
        comps.push_back({0.2, 400 * nm, 20 * nm, 10 * ns, 3000 * ns});
        return comps;
    }
};

class ParticleScintillationTest : public ScintillationTestBase
{
  public:
    //! Create scintillation params
    SPParams build_scintillation_params() override
    {
        ScintillationParams::Input inp;
        inp.resolution_scale.push_back(1);

        // One particle, one component (based on lar-sphere.gdml)
        inp.pid_to_scintpid.push_back(ScintillationParticleId(0));
        ImportParticleScintSpectrum ipss;
        ipss.yield_vector = this->build_particle_yield();
        ipss.components = this->build_particle_components();
        inp.particles.push_back(std::move(ipss));

        return std::make_shared<ScintillationParams>(std::move(inp));
    }

    //! Create particle yield vector
    ImportPhysicsVector build_particle_yield()
    {
        ImportPhysicsVector vec;
        vec.vector_type = ImportPhysicsVectorType::free;
        vec.x = {1e-6, 6};
        vec.y = {3750, 5000};
        return vec;
    }

    //! Create particle components
    VecScintComponents build_particle_components()
    {
        std::vector<ImportScintComponent> vec_comps;
        ImportScintComponent comp;
        comp.yield_frac = 1;
        comp.lambda_mean = from_cm(1e-5);
        comp.lambda_sigma = from_cm(1e-6);
        comp.rise_time = native_value_from(TimeSecond(15e-9));
        comp.fall_time = native_value_from(TimeSecond(5e-9));
        vec_comps.push_back(std::move(comp));
        return vec_comps;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MaterialScintillationTest, data)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    EXPECT_EQ(0, data.num_scint_particles);
    EXPECT_EQ(1, data.materials.size());

    auto const& mat_record = data.materials[opt_mat_];
    EXPECT_REAL_EQ(5, mat_record.yield_per_energy);
    EXPECT_REAL_EQ(1, data.resolution_scale[opt_mat_]);
    EXPECT_EQ(3, data.scint_records.size());

    std::vector<real_type> yield_fracs, lambda_means, lambda_sigmas,
        rise_times, fall_times;
    for (auto comp_idx : range(mat_record.components.size()))
    {
        ScintRecord const& comp
            = data.scint_records[mat_record.components[comp_idx]];
        yield_fracs.push_back(data.reals[mat_record.yield_pdf[comp_idx]]);
        lambda_means.push_back(comp.lambda_mean);
        lambda_sigmas.push_back(comp.lambda_sigma);
        rise_times.push_back(comp.rise_time);
        fall_times.push_back(comp.fall_time);
    }

    real_type norm{0};
    for (auto const& comp : this->build_material_components())
    {
        norm += comp.yield_frac;
    }
    std::vector<real_type> expected_yield_fracs, expected_lambda_means,
        expected_lambda_sigmas, expected_rise_times, expected_fall_times;
    for (auto const& comp : this->build_material_components())
    {
        expected_yield_fracs.push_back(comp.yield_frac / norm);
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
TEST_F(MaterialScintillationTest, pre_generator)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    EXPECT_FALSE(data.scintillation_by_particle());

    // The particle's energy is necessary for the particle track view but is
    // irrelevant for the test since what matters is the energy deposition
    auto particle
        = this->make_particle_track_view(post_energy_, pdg::electron());
    auto const pre_step = this->build_pre_step();

    ScintillationOffload generate(particle,
                                  this->make_sim_track_view(step_length_),
                                  post_pos_,
                                  edep_,
                                  data,
                                  pre_step);

    Rng rng;
    auto const result = generate(rng);
    ASSERT_TRUE(result);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }

    EXPECT_EQ(4, result.num_photons);
    EXPECT_REAL_EQ(0, result.time);
    EXPECT_REAL_EQ(from_cm(step_length_), result.step_length);
    EXPECT_EQ(-1, result.charge.value());
    EXPECT_EQ(0, result.material.get());
    EXPECT_EQ(pre_step.speed.value(),
              result.points[StepPoint::pre].speed.value());
    EXPECT_EQ(particle.speed().value(),
              result.points[StepPoint::post].speed.value());
    EXPECT_VEC_EQ(pre_step.pos, result.points[StepPoint::pre].pos);
    EXPECT_VEC_EQ(post_pos_, result.points[StepPoint::post].pos);
}

//---------------------------------------------------------------------------//
TEST_F(MaterialScintillationTest, basic)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    EXPECT_FALSE(data.scintillation_by_particle());

    auto const pre_step = this->build_pre_step();

    // Pre-generate optical distribution data
    ScintillationOffload generate(
        this->make_particle_track_view(post_energy_, pdg::electron()),
        this->make_sim_track_view(step_length_),
        post_pos_,
        edep_,
        data,
        pre_step);

    Rng rng;
    auto const generated_dist = generate(rng);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }
    auto const inc_dir
        = make_unit_vector(generated_dist.points[StepPoint::post].pos
                           - generated_dist.points[StepPoint::pre].pos);

    // Create the generator and output vectors
    std::vector<Primary> primary_storage(generated_dist.num_photons);
    ScintillationGenerator generate_photons(
        generated_dist, params->host_ref(), make_span(primary_storage));
    std::vector<real_type> energy;
    std::vector<real_type> time;
    std::vector<real_type> cos_theta;
    std::vector<real_type> polarization_x;
    real_type avg_lambda{};
    real_type avg_time{};
    real_type avg_cosine{};
    size_type num_photons{};

    // Generate 2 batches of optical photons from the given input, keep 2
    for ([[maybe_unused]] auto i : range(100))
    {
        auto photons = generate_photons(rng);
        ASSERT_EQ(photons.size(), generated_dist.num_photons);

        // Accumulate averages
        for (Primary const& p : photons)
        {
            avg_lambda += detail::energy_to_wavelength(p.energy);
            avg_time += p.time;
            avg_cosine += dot_product(p.direction, inc_dir);
        }

        if (i < 2)
        {
            // Store individual results
            for (Primary const& p : photons)
            {
                energy.push_back(p.energy.value());
                time.push_back(native_value_to<TimeSecond>(p.time).value());
                cos_theta.push_back(dot_product(p.direction, inc_dir));

                polarization_x.push_back(p.polarization[0]);
                EXPECT_SOFT_EQ(0, dot_product(p.polarization, p.direction));
            }
        }

        num_photons += generated_dist.num_photons;
    }

    avg_lambda = to_cm(avg_lambda / num_photons);
    avg_time = native_value_to<TimeSecond>(avg_time / num_photons).value();
    avg_cosine /= num_photons;

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_EQ(1.9986123389070948e-05, avg_lambda);
        EXPECT_SOFT_EQ(1.13945074214778e-06, avg_time);
        EXPECT_SOFT_EQ(-0.031526689677727655, avg_cosine);
        EXPECT_EQ(7168, rng.exchange_count());

        static double const expected_energy[] = {
            1.2738667300123e-05,
            1.3585330246347e-05,
            6.6061089464883e-06,
            3.3419496205708e-06,
            1.2365331190888e-05,
            1.2985378554366e-05,
            6.3799069497392e-06,
            3.3892880823207e-06,
        };
        static double const expected_time[] = {
            3.211612780853e-08,
            6.1750109166679e-09,
            1.7964384622073e-06,
            6.2340101132549e-06,
            2.0938021428725e-09,
            1.3424826808147e-09,
            2.7672422928171e-06,
            1.29065395794e-05,
        };
        static double const expected_cos_theta[] = {
            0.98576260383561,
            0.27952671419631,
            0.48129448935284,
            0.7448576401346,
            -0.748206733056,
            0.42140775018143,
            0.88014805759581,
            0.6194690974697,
        };
        static double const expected_polarization_x[] = {
            -0.97819537168632,
            0.68933315879807,
            -0.26839376593079,
            -0.45610110755268,
            0.027501392904077,
            0.74278820887819,
            -0.68599121517934,
            0.37271993746494,
        };

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_time, time);
        EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_polarization_x, polarization_x);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ParticleScintillationTest, basic)
{
    GTEST_SKIP() << "particle scintillation is not yet implemented";
}

//---------------------------------------------------------------------------//
TEST_F(MaterialScintillationTest, stress_test)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    auto const pre_step = this->build_pre_step();

    ScintillationOffload generate(
        this->make_particle_track_view(post_energy_, pdg::electron()),
        this->make_sim_track_view(step_length_),
        post_pos_,
        edep_,
        data,
        pre_step);

    Rng rng;
    auto result = generate(rng);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }

    // Overwrite result to force a large number of optical photons
    result.num_photons = 123456;

    // Output data
    std::vector<Primary> storage(result.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(result, data, make_span(storage));

    // Generate optical photons for a given input
    auto photons = generate_photons(rng);

    // Check results
    double avg_lambda{0};
    double hc = constants::h_planck * constants::c_light / units::Mev::value();
    for (auto i : range(result.num_photons))
    {
        avg_lambda += hc / photons[i].energy.value();
    }
    avg_lambda /= static_cast<double>(result.num_photons);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_NEAR(
            16.742580352514256,
            rng.exchange_count() / static_cast<double>(result.num_photons),
            1e-2);
    }

    double expected_lambda{0};
    double expected_error{0};

    auto const& mat_record = data.materials[result.material];
    for (auto comp_idx : range(mat_record.components.size()))
    {
        ScintRecord const& component
            = data.scint_records[mat_record.components[comp_idx]];
        real_type yield = data.reals[mat_record.yield_pdf[comp_idx]];
        expected_lambda += component.lambda_mean * yield;
        expected_error += component.lambda_sigma * yield;
    }
    EXPECT_SOFT_NEAR(avg_lambda, expected_lambda, expected_error);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
