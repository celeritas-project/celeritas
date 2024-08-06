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
#include "celeritas/phys/ParticleParams.hh"

#include "DiagnosticRngEngine.hh"
#include "OpticalTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using namespace celeritas::optical;
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ScintillationTestBase : public OpticalTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using HostValue = HostVal<ScintillationData>;
    using MevEnergy = units::MevEnergy;
    using LightSpeed = units::LightSpeed;
    using SPParams = std::shared_ptr<ScintillationParams>;
    using VecScintComponents = std::vector<ImportScintComponent>;
    //!@}

  protected:
    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    virtual SPParams build_scintillation_params() = 0;

    //! Create particle yield vector
    ImportPhysicsVector build_particle_yield()
    {
        ImportPhysicsVector vec;
        vec.vector_type = ImportPhysicsVectorType::free;
        vec.x = {1e-6, 6};
        vec.y = {3750, 5000};
        return vec;
    }

    //! Set up mock pre-generator step data
    OffloadPreStepData build_pre_step()
    {
        OffloadPreStepData pre_step;
        pre_step.speed = LightSpeed(0.99862874144970537);  // 10 MeV
        pre_step.pos = {0, 0, 0};
        pre_step.time = 0;
        pre_step.opt_mat = opt_mat_;
        return pre_step;
    }

  protected:
    RandomEngine rng_;
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
        comps.push_back({0.65713, 128 * nm, 10 * nm, 10 * ns, 6 * ns});
        comps.push_back({0.31987, 128 * nm, 10 * nm, 0, 1500 * ns});
        comps.push_back({0.023, 200 * nm, 20 * nm, 10 * ns, 3000 * ns});
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

    //! Create particle components
    VecScintComponents build_particle_components()
    {
        constexpr auto cm = units::centimeter;
        constexpr auto sec = units::second;

        std::vector<ImportScintComponent> vec_comps;
        ImportScintComponent comp;
        comp.yield_per_energy = 4000;
        comp.lambda_mean = 1e-5 * cm;
        comp.lambda_sigma = 1e-6 * cm;
        comp.rise_time = 15e-9 * sec;
        comp.fall_time = 5e-9 * sec;
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

    auto const& material = data.materials[opt_mat_];
    EXPECT_REAL_EQ(5, material.yield_per_energy);
    EXPECT_REAL_EQ(1, data.resolution_scale[opt_mat_]);
    EXPECT_EQ(3, data.components.size());

    std::vector<real_type> yield_fracs, lambda_means, lambda_sigmas,
        rise_times, fall_times;
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
    for (auto const& comp : this->build_material_components())
    {
        norm += comp.yield_per_energy;
    }
    std::vector<real_type> expected_yield_fracs, expected_lambda_means,
        expected_lambda_sigmas, expected_rise_times, expected_fall_times;
    for (auto const& comp : this->build_material_components())
    {
        expected_yield_fracs.push_back(comp.yield_per_energy / norm);
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
TEST_F(ParticleScintillationTest, data)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    EXPECT_TRUE(data.scintillation_by_particle());

    auto const scint_pid = data.pid_to_scintpid[ParticleId{0}];
    EXPECT_EQ(1, data.pid_to_scintpid.size());
    EXPECT_EQ(1, data.num_scint_particles);
    EXPECT_REAL_EQ(1, data.resolution_scale[opt_mat_]);

    // Get correct spectrum index given opticals particle and material ids
    auto const part_scint_spectrum_id
        = data.spectrum_index(scint_pid, opt_mat_);
    EXPECT_EQ(0, part_scint_spectrum_id.get());

    auto const& particle = data.particles[part_scint_spectrum_id];
    EXPECT_EQ(particle.yield_vector.grid.size(),
              particle.yield_vector.value.size());

    std::vector<real_type> yield_grid, yield_value;
    for (auto i : range(particle.yield_vector.grid.size()))
    {
        auto grid_idx = particle.yield_vector.grid[i];
        auto val_idx = particle.yield_vector.value[i];
        yield_grid.push_back(data.reals[grid_idx]);
        yield_value.push_back(data.reals[val_idx]);
    }

    std::vector<real_type> yield_fracs, lambda_means, lambda_sigmas,
        rise_times, fall_times;
    for (auto i : range(particle.components.size()))
    {
        auto comp_idx = particle.components[i];
        yield_fracs.push_back(data.components[comp_idx].yield_frac);
        lambda_means.push_back(data.components[comp_idx].lambda_mean);
        lambda_sigmas.push_back(data.components[comp_idx].lambda_sigma);
        rise_times.push_back(data.components[comp_idx].rise_time);
        fall_times.push_back(data.components[comp_idx].fall_time);
    }

    // Particle yield vector
    static double const expected_yield_grid[] = {1e-06, 6};
    static double const expected_yield_value[] = {3750, 5000};

    EXPECT_VEC_SOFT_EQ(expected_yield_grid, yield_grid);
    EXPECT_VEC_SOFT_EQ(expected_yield_value, yield_value);

    // Particle components
    static double const expected_yield_fracs[] = {1};
    static double const expected_lambda_means[] = {1e-05};
    static double const expected_lambda_sigmas[] = {1e-06};
    static double const expected_rise_times[] = {1.5e-08};
    static double const expected_fall_times[] = {5e-09};

    EXPECT_VEC_SOFT_EQ(expected_yield_fracs, yield_fracs);
    EXPECT_VEC_SOFT_EQ(expected_lambda_means, lambda_means);
    EXPECT_VEC_SOFT_EQ(expected_lambda_sigmas, lambda_sigmas);
    EXPECT_VEC_SOFT_EQ(expected_rise_times, rise_times);
    EXPECT_VEC_SOFT_EQ(expected_fall_times, fall_times);
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

    auto const result = generate(this->rng());
    CELER_ASSERT(result);

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

    auto const generated_dist = generate(this->rng());
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

    // Generate 2 batches of optical photons from the given input
    for ([[maybe_unused]] auto i : range(2))
    {
        auto photons = generate_photons(this->rng());
        ASSERT_EQ(photons.size(), generated_dist.num_photons);

        for (Primary const& p : photons)
        {
            energy.push_back(p.energy.value());
            time.push_back(p.time / units::second);
            cos_theta.push_back(dot_product(p.direction, inc_dir));

            polarization_x.push_back(p.polarization[0]);
            EXPECT_SOFT_EQ(0, dot_product(p.polarization, p.direction));
        }
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_energy[] = {
            1.0108118605375e-05,
            1.1217590386333e-05,
            1.0717754890017e-05,
            7.2508084886886e-06,
            9.645934625422e-06,
            1.0422369961991e-05,
            1.0134774090994e-05,
            7.4773995536571e-06,
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

    auto result = generate(this->rng());

    // Overwrite result to force a large number of optical photons
    result.num_photons = 123456;

    // Output data
    std::vector<Primary> storage(result.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(result, data, make_span(storage));

    // Generate optical photons for a given input
    auto photons = generate_photons(this->rng());

    // Check results
    double avg_lambda{0};
    double hc = constants::h_planck * constants::c_light / units::Mev::value();
    for (auto i : range(result.num_photons))
    {
        avg_lambda += hc / photons[i].energy.value();
    }
    avg_lambda /= static_cast<double>(result.num_photons);

    double expected_lambda{0};
    double expected_error{0};

    for (auto i : data.materials[result.material].components)
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
