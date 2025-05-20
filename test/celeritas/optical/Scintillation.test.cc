//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Scintillation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"
#include "celeritas/optical/gen/GeneratorDistributionData.hh"
#include "celeritas/optical/gen/ScintillationData.hh"
#include "celeritas/optical/gen/ScintillationGenerator.hh"
#include "celeritas/optical/gen/ScintillationOffload.hh"
#include "celeritas/optical/gen/ScintillationParams.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "OpticalTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using celeritas::test::from_cm;
using celeritas::test::to_cm;
using TimeSecond = celeritas::RealQuantity<celeritas::units::Second>;

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
    OptMatId opt_mat_{0};

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
        static constexpr real_type nm{units::meter * 1e-9};
        static constexpr real_type ns{units::nanosecond};

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
        inp.pid_to_scintpid.push_back(ScintParticleId(0));
        ImportParticleScintSpectrum ipss;
        ipss.yield_vector = this->build_particle_yield();
        ipss.components = this->build_particle_components();
        inp.particles.push_back(std::move(ipss));

        return std::make_shared<ScintillationParams>(std::move(inp));
    }

    //! Create particle yield vector
    inp::Grid build_particle_yield()
    {
        inp::Grid vec;
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
    ScintillationGenerator generate_photon(params->host_ref(), generated_dist);
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
        for ([[maybe_unused]] auto j : range(generated_dist.num_photons))
        {
            auto p = generate_photon(rng);

            // Accumulate averages
            avg_lambda += optical::detail::energy_to_wavelength(p.energy);
            avg_time += p.time;
            avg_cosine += dot_product(p.direction, inc_dir);

            if (i < 2)
            {
                // Store individual results
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
        EXPECT_SOFT_EQ(1.8023146707476483e-05, avg_lambda);
        EXPECT_SOFT_EQ(8.6510374107600554e-07, avg_time);
        EXPECT_SOFT_EQ(-0.0078894853694884293, avg_cosine);
        EXPECT_EQ(7602, rng.exchange_count());

        static double const expected_energy[] = {
            6.1650902874689e-06,
            6.1852526228383e-06,
            6.6524813707218e-06,
            1.2141478183957e-05,
            1.221301636759e-05,
            5.8200972038835e-06,
            1.2759813899478e-05,
            1.2232069181772e-05,
        };
        static double const expected_time[] = {
            3.3128806993047e-06,
            1.9448090540859e-07,
            1.1174848154165e-06,
            1.2460198181058e-08,
            3.5306344404732e-08,
            3.19537294006e-07,
            7.2757167500751e-09,
            3.5272895177539e-09,
        };
        static double const expected_cos_theta[] = {
            0.99292265109602,
            -0.4059411008841,
            -0.57615133521653,
            -0.65226965599904,
            -0.08402168914221,
            -0.087934351005127,
            0.88014805759581,
            0.81472943553235,
        };
        static double const expected_polarization_x[] = {
            -0.48061717648891,
            0.37029605368662,
            0.78751570900663,
            -0.39528947901676,
            0.019773814327391,
            0.95928367243846,
            -0.68599121517934,
            -0.35306899564942,
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

    // Generate optical photons for a given input
    Rng rng;
    auto result = generate(rng);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }

    // Create the generator
    ScintillationGenerator generate_photon(data, result);

    // Check results
    real_type avg_lambda{0};
    int const num_photons{123456};
    for ([[maybe_unused]] auto i : range(num_photons))
    {
        auto p = generate_photon(rng);
        avg_lambda += optical::detail::energy_to_wavelength(p.energy);
    }
    avg_lambda /= static_cast<real_type>(num_photons);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_NEAR(
            18.724841238983931,
            rng.exchange_count() / static_cast<real_type>(num_photons),
            1e-2);
    }

    real_type expected_lambda{0};
    real_type expected_error{0};

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
}  // namespace celeritas
