//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SeltzerBerger.test.cc
//---------------------------------------------------------------------------//
#include "base/Algorithms.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "io/SeltzerBergerReader.hh"
#include "physics/base/CutoffView.hh"
#include "physics/base/Units.hh"
#include "physics/em/BremsstrahlungProcess.hh"
#include "physics/em/CombinedBremModel.hh"
#include "physics/em/detail/CombinedBremInteractor.hh"
#include "physics/em/detail/RBDiffXsCalculator.hh"
#include "physics/em/detail/SBEnergyDistribution.hh"
#include "physics/em/detail/SBPositronXsCorrector.hh"
#include "physics/material/MaterialTrackView.hh"
#include "physics/material/MaterialView.hh"

#include "../InteractionIO.hh"
#include "../InteractorHostTestBase.hh"
#include "celeritas_test.hh"
#include "gtest/Main.hh"

using celeritas::BremsstrahlungProcess;
using celeritas::CombinedBremModel;
using celeritas::ElementComponentId;
using celeritas::ElementId;
using celeritas::ElementView;
using celeritas::SeltzerBergerReader;
using celeritas::detail::CombinedBremInteractor;
using celeritas::detail::RBDiffXsCalculator;
using celeritas::detail::SBElectronXsCorrector;
using celeritas::detail::SBEnergyDistHelper;
using celeritas::detail::SBEnergyDistribution;
using celeritas::detail::SBPositronXsCorrector;

using celeritas::units::AmuMass;
using celeritas::units::MevMass;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

using Energy   = celeritas::units::MevEnergy;
using EnergySq = SBEnergyDistHelper::EnergySq;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CombinedBremTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::MatterState;
        using celeritas::ParticleRecord;
        using namespace celeritas::constants;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleRecord::stable_decay_constant();

        // Set up shared particle data
        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"positron",
              pdg::positron(),
              MevMass{0.5109989461},
              ElementaryCharge{1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});
        const auto& particles = *this->particle_params();

        data_.rb_data.ids.electron = particles.find(pdg::electron());
        data_.rb_data.ids.positron = particles.find(pdg::positron());
        data_.rb_data.ids.gamma    = particles.find(pdg::gamma());
        data_.rb_data.electron_mass
            = particles.get(data_.rb_data.ids.electron).mass();

        // Set default particle to incident 1 MeV photon
        this->set_inc_particle(pdg::electron(), MevEnergy{1.0});
        this->set_inc_direction({0, 0, 1});

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        mat_inp.materials = {
            {0.141 * na_avogadro,
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);
        this->set_material("Cu");

        // Set up Seltzer-Berger cross section data
        std::string         data_path = this->test_data_path("physics/em", "");
        SeltzerBergerReader read_element_data(data_path.c_str());

        // Construct SeltzerBergerModel and set host data
        model_ = std::make_shared<CombinedBremModel>(ModelId{0},
                                                     *this->particle_params(),
                                                     *this->material_params(),
                                                     read_element_data,
                                                     true);
        data_  = model_->host_ref();

        // Set cutoffs
        CutoffParams::Input           input;
        CutoffParams::MaterialCutoffs material_cutoffs;
        material_cutoffs.push_back({MevEnergy{0.02064384}, 0.07});
        input.materials = this->material_params();
        input.particles = this->particle_params();
        input.cutoffs.insert({pdg::gamma(), material_cutoffs});
        this->set_cutoff_params(input);
    }

    EnergySq density_correction(MaterialId matid, Energy e) const
    {
        CELER_EXPECT(matid);
        CELER_EXPECT(e > celeritas::zero_quantity());
        using celeritas::ipow;
        using namespace celeritas::constants;

        auto           mat    = this->material_params()->get(matid);
        constexpr auto migdal = 4 * pi * r_electron
                                * ipow<2>(lambdabar_electron);

        real_type density_factor = mat.electron_density() * migdal;
        return EnergySq{density_factor * ipow<2>(e.value())};
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);
    }

  protected:
    std::shared_ptr<CombinedBremModel>       model_;
    celeritas::detail::CombinedBremRef       data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CombinedBremTest, basic_seltzer_berger)
{
    using celeritas::MaterialView;

    // Reserve 4 secondaries, one for each sample
    const int num_samples = 4;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().make_material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Create the interactor
    CombinedBremInteractor interact(model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    cutoffs,
                                    this->secondary_allocator(),
                                    material_view,
                                    ElementComponentId{0});
    RandomEngine&          rng_engine = this->rng();

    // Produce two samples from the original/incident photon
    std::vector<double> angle;
    std::vector<double> energy;

    // Loop number of samples
    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(celeritas::dot_product(
            result.direction, result.secondaries.front().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    const double expected_angle[] = {0.959441513277674,
                                     0.994350429950924,
                                     0.968866136008621,
                                     0.961582855967571};

    const double expected_energy[] = {0.0349225070114679,
                                      0.0316182310804369,
                                      0.0838794010486177,
                                      0.106195186929141};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(CombinedBremTest, basic_relativistic_brem)
{
    const int num_samples = 4;

    // Reserve  num_samples secondaries;
    this->resize_secondaries(num_samples);

    // Production cuts
    auto material_view = this->material_track().make_material_view();
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});

    // Set the incident particle energy
    this->set_inc_particle(pdg::electron(), MevEnergy{25000});

    // Create the interactor
    CombinedBremInteractor interact(model_->host_ref(),
                                    this->particle_track(),
                                    this->direction(),
                                    cutoffs,
                                    this->secondary_allocator(),
                                    material_view,
                                    ElementComponentId{0});

    RandomEngine& rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<double> angle;
    std::vector<double> energy;

    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        energy.push_back(result.secondaries[0].energy.value());
        angle.push_back(celeritas::dot_product(
            result.direction, result.secondaries.back().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.

    const double expected_energy[] = {
        18844.5999305425, 42.185863858534, 3991.9107959354, 212.273682952066};

    const double expected_angle[] = {0.999999972054405,
                                     0.999999999587026,
                                     0.999999999684891,
                                     0.999999999474844};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(CombinedBremTest, stress_test_combined)
{
    using celeritas::MaterialView;

    const int           num_samples = 1e4;
    std::vector<double> avg_engine_samples;
    std::vector<double> avg_energy_samples;

    // Views
    auto cutoffs       = this->cutoff_params()->get(MaterialId{0});
    auto material_view = this->material_track().make_material_view();

    // Loop over a set of incident gamma energies
    const real_type test_energy[]
        = {1.5, 5, 10, 50, 100, 1000, 1e+4, 1e+5, 1e+6};

    for (auto particle : {pdg::electron(), pdg::positron()})
    {
        for (real_type inc_e : test_energy)
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            //            this->set_inc_particle(particle, MevEnergy{inc_e});

            RandomEngine&           rng_engine            = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;
            double                  tot_energy_sampled    = 0;

            // Loop over several incident directions
            for (const Real3& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                this->set_inc_particle(particle, MevEnergy{inc_e});
                CombinedBremInteractor interact(model_->host_ref(),
                                                this->particle_track(),
                                                this->direction(),
                                                cutoffs,
                                                this->secondary_allocator(),
                                                material_view,
                                                ElementComponentId{0});

                // Loop over many particles
                for (unsigned int i = 0; i < num_samples; ++i)
                {
                    Interaction result = interact(rng_engine);
                    this->sanity_check(result);
                    tot_energy_sampled += result.secondaries[0].energy.value();
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng_engine.count())
                                         / double(num_particles_sampled));
            avg_energy_samples.push_back(tot_energy_sampled
                                         / double(num_particles_sampled));
        }
    }

    // Gold values for average number of calls to RNG
    static const double expected_avg_engine_samples[] = {14.088,
                                                         13.2402,
                                                         12.9641,
                                                         12.5832,
                                                         12.4988,
                                                         12.3433,
                                                         12.4378,
                                                         13.2556,
                                                         15.3633,
                                                         14.2366,
                                                         13.2618,
                                                         12.93,
                                                         12.5757,
                                                         12.5079,
                                                         12.3337,
                                                         12.4192,
                                                         13.293,
                                                         15.3786};

    static const double expected_avg_energy_samples[] = {0.20338654094171,
                                                         0.53173619503507,
                                                         0.99638562846318,
                                                         4.4359411867158,
                                                         8.7590072534526,
                                                         85.185116736899,
                                                         905.94487251514,
                                                         10719.081816783,
                                                         149600.77957549,
                                                         0.18898378115149,
                                                         0.52241408503454,
                                                         0.98804532103107,
                                                         4.4248220614854,
                                                         8.4970015150131,
                                                         85.409453635372,
                                                         918.52583614877,
                                                         10756.648855071,
                                                         146921.82188051};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy_samples, avg_energy_samples);
}
