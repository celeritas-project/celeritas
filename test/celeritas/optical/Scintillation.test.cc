//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Scintillation.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/ParamsDataStore.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/TrackInitializer.hh"
#include "celeritas/optical/detail/OpticalUtils.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
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
        pre_step.speed = LightSpeed{0.9988175606678128};  // 10 MeV
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

class MaterialScintillationGaussianTest : public ScintillationTestBase
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
        comps.push_back({0.5, 10 * ns, 6 * ns, {100 * nm, 5 * nm}, {}});
        comps.push_back({0.3, 0, 1500 * ns, {200 * nm, 10 * nm}, {}});
        comps.push_back({0.2, 10 * ns, 3000 * ns, {400 * nm, 20 * nm}, {}});

        return comps;
    }
};

class MaterialScintillationTabularTest : public ScintillationTestBase
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
        static constexpr real_type ns{units::nanosecond};

        // Note these components are in tabular form
        std::vector<ImportScintComponent> comps;
        comps.push_back(
            {0.2, 10 * ns, 1500 * ns, {}, {{1.0, 2.0, 3.0}, {0.5, 0.3, 0.2}}});

        return comps;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MaterialScintillationGaussianTest, data)
{
    auto const params = this->build_scintillation_params();
    EXPECT_FALSE(params->is_geant_compatible());
    auto const& data = params->host_ref();

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
        expected_lambda_means.push_back(comp.gauss.lambda_mean);
        expected_lambda_sigmas.push_back(comp.gauss.lambda_sigma);
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
TEST_F(MaterialScintillationGaussianTest, pre_generator)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    // The particle's energy is necessary for the particle track view but
    // is irrelevant for the test since what matters is the energy
    // deposition
    auto particle
        = this->make_particle_track_view(post_energy_, pdg::electron());
    auto const pre_step = this->build_pre_step();
    auto sim = this->make_sim_track_view(step_length_);
    sim.add_time(sim.step_length() / native_value_from(particle.speed()));
    OffloadPrePostStepData pre_post_step{particle.speed(), edep_};

    ScintillationOffload generate(
        particle, sim, post_pos_, edep_, data, pre_step, pre_post_step);

    Rng rng;
    auto const result = generate(rng);
    ASSERT_TRUE(result);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }

    EXPECT_EQ(4, result.num_photons);
    EXPECT_REAL_EQ(from_cm(step_length_), result.step_length);
    EXPECT_EQ(-1, result.charge.value());
    EXPECT_EQ(0, result.material.get());
    EXPECT_EQ(pre_step.speed.value(),
              result.points[StepPoint::pre].speed.value());
    EXPECT_EQ(particle.speed().value(),
              result.points[StepPoint::post].speed.value());
    EXPECT_EQ(pre_step.time, result.points[StepPoint::pre].time);
    EXPECT_EQ(sim.time(), result.points[StepPoint::post].time);
    EXPECT_VEC_EQ(pre_step.pos, result.points[StepPoint::pre].pos);
    EXPECT_VEC_EQ(post_pos_, result.points[StepPoint::post].pos);
}

//---------------------------------------------------------------------------//
TEST_F(MaterialScintillationGaussianTest, basic)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    auto particle
        = this->make_particle_track_view(post_energy_, pdg::electron());
    auto sim = this->make_sim_track_view(step_length_);
    auto const pre_step = this->build_pre_step();
    OffloadPrePostStepData pre_post_step{particle.speed(), edep_};

    // Pre-generate optical distribution data
    ScintillationOffload generate(
        particle, sim, post_pos_, edep_, data, pre_step, pre_post_step);

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
    optical::ScintillationGenerator generate_photon(params->host_ref(),
                                                    generated_dist);
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
                time.push_back(p.time / units::nanosecond);
                cos_theta.push_back(dot_product(p.direction, inc_dir));

                polarization_x.push_back(p.polarization[0]);
                EXPECT_SOFT_EQ(0, dot_product(p.polarization, p.direction));
            }
        }

        num_photons += generated_dist.num_photons;
    }

    avg_lambda = to_cm(avg_lambda / num_photons);
    avg_time = avg_time / (units::nanosecond * num_photons);
    avg_cosine /= num_photons;

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_EQ(1.8455823106866e-05, avg_lambda);
        EXPECT_SOFT_EQ(869.57091905169, avg_time);
        EXPECT_SOFT_EQ(0.023544714177305, avg_cosine);
        EXPECT_EQ(8250, rng.exchange_count());

        static double const expected_energy[] = {
            6.1650902874689e-06,
            5.3609823654525e-06,
            1.1892383769681e-05,
            6.0546751413336e-06,
            1.3582229285385e-05,
            6.1392669704928e-06,
            3.1773246015197e-06,
            2.9038196053521e-06,
        };
        static double const expected_time[] = {
            3312.8806914137,
            338.64626300081,
            10.532092321203,
            404.28257965362,
            35.26357244485,
            295.4003690722,
            4407.0565774611,
            138.14745536841,
        };
        static double const expected_cos_theta[] = {
            0.99292265109602,
            -0.77507096788764,
            -0.20252296017542,
            -0.70177204718526,
            -0.57958185096199,
            0.14750933319318,
            -0.15366976713254,
            -0.97292174666956,
        };
        static double const expected_polarization_x[] = {
            -0.48061717648891,
            0.74609610658139,
            0.99419460248005,
            -0.57457399792055,
            0.5101014413042,
            0.30947392565286,
            0.11400602132643,
            -0.47137697798179,
        };
        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_time, time);
        EXPECT_VEC_SOFT_EQ(expected_cos_theta, cos_theta);
        EXPECT_VEC_SOFT_EQ(expected_polarization_x, polarization_x);
    }
}

//---------------------------------------------------------------------------//
TEST_F(MaterialScintillationGaussianTest, time)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    optical::GeneratorDistributionData gdd;
    gdd.type = GeneratorType::scintillation;
    gdd.num_photons = 8;
    gdd.step_length = from_cm(step_length_);
    gdd.charge = units::ElementaryCharge{-1};
    gdd.material = opt_mat_;
    gdd.points[StepPoint::pre].pos = {0, 0, 0};
    gdd.points[StepPoint::post].pos = post_pos_;

    auto sample_time = [&](optical::GeneratorDistributionData const& d) {
        Rng rng;
        optical::ScintillationGenerator generate(data, d);
        std::vector<real_type> time;
        for (size_type i = 0; i < d.num_photons; ++i)
        {
            auto p = generate(rng);
            time.push_back(p.time / units::nanosecond);
        }
        return time;
    };

    // Use pre- and post-step time to sample time
    {
        auto particle
            = this->make_particle_track_view(post_energy_, pdg::electron());
        gdd.points[StepPoint::pre].time = 0;
        gdd.points[StepPoint::post].time
            = from_cm(step_length_) / native_value_from(particle.speed());

        auto time = sample_time(gdd);

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            static double const expected_time[] = {
                7.3670494614798,
                10.58035645366,
                3117.0542454245,
                4968.1642964938,
                2.1742646282647,
                319.54758544355,
                3.3419800716698,
                6453.3407764283,
            };
            EXPECT_VEC_SOFT_EQ(expected_time, time);
        }
    }

    // Use pre- and post-step speed to sample time
    {
        auto particle
            = this->make_particle_track_view(post_energy_, pdg::electron());
        gdd.points[StepPoint::pre].speed = this->build_pre_step().speed;
        gdd.points[StepPoint::post].speed = particle.speed();

        auto time = sample_time(gdd);

        if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            static double const expected_time[] = {
                7.367041567676,
                10.580348559856,
                3117.0542375307,
                4968.1642886,
                2.1742567344609,
                319.54757754975,
                3.341972177866,
                6453.3407685345,
            };
            EXPECT_VEC_SOFT_EQ(expected_time, time);
        }
    }
}

//---------------------------------------------------------------------------//
TEST_F(MaterialScintillationGaussianTest, stress_test)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    auto particle
        = this->make_particle_track_view(post_energy_, pdg::electron());
    auto const pre_step = this->build_pre_step();
    OffloadPrePostStepData pre_post_step{particle.speed(), edep_};

    ScintillationOffload generate(particle,
                                  this->make_sim_track_view(step_length_),
                                  post_pos_,
                                  edep_,
                                  data,
                                  pre_step,
                                  pre_post_step);

    // Generate optical photons for a given input
    Rng rng;
    auto result = generate(rng);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_EQ(10, rng.exchange_count());
    }

    // Create the generator
    optical::ScintillationGenerator generate_photon(data, result);

    // Check results
    real_type avg_lambda{0};
    int const num_photons{10000};
    for ([[maybe_unused]] auto i : range(num_photons))
    {
        auto p = generate_photon(rng);
        avg_lambda += optical::detail::energy_to_wavelength(p.energy);
    }
    avg_lambda /= static_cast<real_type>(num_photons);
    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        EXPECT_SOFT_NEAR(
            20.71518597719,
            rng.exchange_count() / static_cast<real_type>(num_photons),
            1e-2);
    }

    real_type expected_lambda{0};

    auto const& mat_record = data.materials[result.material];
    for (auto comp_idx : range(mat_record.components.size()))
    {
        ScintRecord const& component
            = data.scint_records[mat_record.components[comp_idx]];
        real_type yield = data.reals[mat_record.yield_pdf[comp_idx]];
        expected_lambda += component.lambda_mean * yield;
    }
    EXPECT_SOFT_NEAR(avg_lambda, expected_lambda, 1e-4);
}

TEST_F(MaterialScintillationTabularTest, uses_nonuniform_grid_calculator)
{
    auto const params = this->build_scintillation_params();
    EXPECT_TRUE(params->is_geant_compatible());
    auto const& data = params->host_ref();

    // Iterate components and, when an energy CDF is present, construct grid
    for (auto i : range(data.scint_records.size()))
    {
        auto const& rec = data.scint_records[ItemId<ScintRecord>(i)];

        if (rec.energy_cdf)
        {
            NonuniformGridCalculator calc_cdf(data.energy_cdfs[rec.energy_cdf],
                                              data.reals);
            auto const& cdf_grid = calc_cdf.grid();

            EXPECT_SOFT_EQ(1, cdf_grid.front());
            EXPECT_SOFT_EQ(3, cdf_grid.back());

            // Invert a representative CDF value
            auto calc_energy = calc_cdf.make_inverse();
            EXPECT_SOFT_EQ(cdf_grid.front(), calc_energy(0));
            EXPECT_SOFT_EQ(cdf_grid.back(), calc_energy(1));

            std::vector<real_type> energy;
            Rng rng;
            for ([[maybe_unused]] auto i : range(4))
            {
                energy.push_back(calc_energy(generate_canonical(rng)));
            }
            if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
            {
                static double const expected_energy[] = {1.22015013198227,
                                                         2.57102233398591,
                                                         2.919056204923,
                                                         1.3591803198469};

                EXPECT_VEC_SOFT_EQ(expected_energy, energy);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
