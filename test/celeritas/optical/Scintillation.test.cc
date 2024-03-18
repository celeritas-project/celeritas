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
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalPrimary.hh"
#include "celeritas/optical/ScintillationData.hh"
#include "celeritas/optical/ScintillationGenerator.hh"
#include "celeritas/optical/ScintillationParams.hh"
#include "celeritas/optical/ScintillationPreGenerator.hh"
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
    using MevEnergy = units::MevEnergy;
    using LightSpeed = units::LightSpeed;
    //!@}

  protected:
    void SetUp() override {}

    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    //! Create scintillation params
    std::shared_ptr<ScintillationParams>
    build_scintillation_params(bool scint_by_particle = false)
    {
        ImportScintData data;
        data.resolution_scale = 1;
        // One material, three components
        data.material.yield = 5;
        data.material.components = this->build_material_components();
        // One particle, one component (based on lar-sphere.gdml)
        ImportParticleScintSpectrum ipss;
        ipss.yield_vector = this->build_particle_yield();
        ipss.components = this->build_particle_components();
        data.particles.insert({pdg::electron().get(), std::move(ipss)});

        ScintillationParams::Input inp;
        inp.scintillation_by_particle = scint_by_particle;
        inp.matid_to_optmatid.push_back(OpticalMaterialId(0));
        // Match particle params list
        inp.pid_to_scintpid.push_back(ScintillationParticleId(0));  // electron
        inp.data.push_back(std::move(data));

        return std::make_shared<ScintillationParams>(std::move(inp),
                                                     this->particle_params());
    }

    //! Create material components
    std::vector<ImportScintComponent> build_material_components()
    {
        std::vector<ImportScintComponent> comps;
        comps.push_back({0.65713, 128 * nm, 10 * nm, 10 * ns, 6 * ns});
        comps.push_back({0.31987, 128 * nm, 10 * nm, 10 * ns, 1500 * ns});
        comps.push_back({0.023, 200 * nm, 20 * nm, 10 * ns, 3000 * ns});
        return comps;
    }

    //! Create particle yield vector
    ImportPhysicsVector build_particle_yield()
    {
        ImportPhysicsVector vec;
        vec.vector_type = ImportPhysicsVectorType::free;
        vec.x.push_back(1e-6);
        vec.x.push_back(6);
        vec.y.push_back(3750);
        vec.y.push_back(5000);
        return vec;
    }

    //! Create particle components
    std::vector<ImportScintComponent> build_particle_components()
    {
        std::vector<ImportScintComponent> vec_comps;
        ImportScintComponent comp;
        comp.yield = 4000;
        comp.lambda_mean = 1e-5;
        comp.lambda_sigma = 1e-6;
        comp.rise_time = 15 * ns;
        comp.fall_time = 5 * ns;
        vec_comps.push_back(std::move(comp));
        return vec_comps;
    }

    //! Set up mock pre-generator step data
    ScintillationPreGenerator::OpticalPreGenStepData build_pregen_step()
    {
        ScintillationPreGenerator::OpticalPreGenStepData pregen_data;
        pregen_data.energy_dep = MevEnergy{0.75};
        pregen_data.points[StepPoint::pre].speed = LightSpeed(0.99);
        pregen_data.points[StepPoint::post].speed = LightSpeed(0.99 * 0.9);
        pregen_data.points[StepPoint::pre].pos = {0, 0, 0};
        pregen_data.points[StepPoint::post].pos = {0, 0, 1};
        return pregen_data;
    }

  protected:
    RandomEngine rng_;
    static constexpr double nm = 1e-9 * units::meter;
    static constexpr double ns = 1e-9 * units::second;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ScintillationTest, material_scint_params)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    auto const opt_matid = data.matid_to_optmatid[MaterialId{0}];

    EXPECT_EQ(1, data.num_materials);
    EXPECT_EQ(1, data.num_particles);
    EXPECT_EQ(1, data.materials.size());
    EXPECT_EQ(0, opt_matid.get());

    auto const& material = data.materials[opt_matid];
    EXPECT_REAL_EQ(5, material.yield);
    EXPECT_REAL_EQ(1, data.resolution_scale[opt_matid]);
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
        norm += comp.yield;
    }
    std::vector<real_type> expected_yield_fracs, expected_lambda_means,
        expected_lambda_sigmas, expected_rise_times, expected_fall_times;
    for (auto const& comp : this->build_material_components())
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
TEST_F(ScintillationTest, particle_scint_params)
{
    auto const params = this->build_scintillation_params(
        /* scint_by_particle = */ true);
    auto const& data = params->host_ref();
    auto const opt_matid = data.matid_to_optmatid[MaterialId{0}];
    auto const scint_pid = data.pid_to_scintpid[ParticleId{0}];

    EXPECT_EQ(1, data.matid_to_optmatid.size());
    EXPECT_EQ(1, data.pid_to_scintpid.size());
    EXPECT_REAL_EQ(1, data.resolution_scale[opt_matid]);

    // Get correct spectrum index given opticals particle and material ids
    auto const part_scint_spectrum_id
        = data.spectrum_index(scint_pid, opt_matid);
    EXPECT_EQ(0, part_scint_spectrum_id.get());

    auto const& particle = data.particles[part_scint_spectrum_id];
    EXPECT_EQ(particle.yield_vector.grid.size(),
              particle.yield_vector.value.size());

    std::vector<real_type> yield_grid, yield_value;
    for (auto i : range(particle.yield_vector.grid.size()))
    {
        auto grid_idx = particle.yield_vector.grid[i];
        auto val_idx = particle.yield_vector.value[i];
        yield_grid.push_back(data.grid_data[grid_idx]);
        yield_value.push_back(data.grid_data[val_idx]);
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
TEST_F(ScintillationTest, pre_generator)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();

    // The particle's energy is necessary for the particle track view but is
    // irrelevant for the test since what matters is the energy deposition,
    // which is hardcoded in this->build_pregen_step()
    ScintillationPreGenerator generate(
        this->make_particle_track_view(MevEnergy{10}, pdg::electron()),
        this->make_sim_track_view(1),
        data.matid_to_optmatid[MaterialId{0}],
        data,
        this->build_pregen_step());

    auto result = generate(this->rng());
    EXPECT_EQ(4, result.num_photons);
    EXPECT_REAL_EQ(0, result.time);
    EXPECT_REAL_EQ(1, result.step_length);
    EXPECT_EQ(-1, result.charge.value());
    EXPECT_EQ(0, result.material.get());

    auto expected_step = this->build_pregen_step();
    for (auto p : range(StepPoint::size_))
    {
        EXPECT_EQ(expected_step.points[p].speed.value(),
                  result.points[p].speed.value());
        EXPECT_VEC_EQ(expected_step.points[p].pos, result.points[p].pos);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ScintillationTest, basic)
{
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    auto const opt_matid = data.matid_to_optmatid[MaterialId{0}];

    // Pre-generate optical distribution data
    ScintillationPreGenerator generate(
        this->make_particle_track_view(MevEnergy{10}, pdg::electron()),
        this->make_sim_track_view(1),
        opt_matid,
        data,
        this->build_pregen_step());

    auto result = generate(this->rng());

    // Output data
    std::vector<OpticalPrimary> storage(result.num_photons);

    // Create the generator
    ScintillationGenerator generate_photons(
        result, params->host_ref(), make_span(storage));
    RandomEngine& rng_engine = this->rng();

    // Generate optical photons for a given input
    auto photons = generate_photons(rng_engine);

    // Check results
    std::vector<real_type> energy, time, cos_theta, polarization_x, cos_polar;
    for (auto i : range(result.num_photons))
    {
        energy.push_back(photons[i].energy.value());
        time.push_back(photons[i].time / units::second);
        cos_theta.push_back(
            dot_product(photons[i].direction,
                        result.points[StepPoint::post].pos
                            - result.points[StepPoint::pre].pos));
        polarization_x.push_back(photons[i].polarization[0]);
        cos_polar.push_back(
            dot_product(photons[i].polarization, photons[i].direction));
    }

    if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
    {
        static double const expected_energy[] = {1.0108118605375e-05,
                                                 1.1217590386333e-05,
                                                 1.0717754890017e-05,
                                                 5.9167264717999e-06};
        static double const expected_time[] = {3.2080893159083e-08,
                                               6.136381528505e-09,
                                               1.7964298529751e-06,
                                               8.0854850049769e-07};
        static double const expected_cos_theta[] = {0.98576260383561,
                                                    0.27952671419631,
                                                    0.48129448935284,
                                                    -0.70177204718526};
        static double const expected_polarization_x[] = {-0.97819537168632,
                                                         0.68933315879807,
                                                         -0.26839376593079,
                                                         -0.57457399792055};
        static double const expected_cos_polar[] = {0, 0, 0, 0};

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
    auto const params = this->build_scintillation_params();
    auto const& data = params->host_ref();
    auto const opt_matid = data.matid_to_optmatid[MaterialId{0}];

    ScintillationPreGenerator generate(
        this->make_particle_track_view(MevEnergy{10}, pdg::electron()),
        this->make_sim_track_view(1),
        opt_matid,
        data,
        this->build_pregen_step());
    auto result = generate(this->rng());

    // Overwrite result to force a large number of optical photons
    result.num_photons = 123456;

    // Output data
    std::vector<OpticalPrimary> storage(result.num_photons);

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
