//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MuHadIonization.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/MuHadIonizationInteractor.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/InteractionIO.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MuHadIonizationTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using namespace units;

        // Set up shared material data
        MaterialParams::Input mat_inp;
        mat_inp.elements = {{AtomicNumber{29}, AmuMass{63.546}, {}, "Cu"}};
        mat_inp.materials = {
            {native_value_from(MolCcDensity{0.141}),
             293.0,
             MatterState::solid,
             {{ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(mat_inp);

        // Set 1 keV electron cutoff
        CutoffParams::Input cut_inp;
        cut_inp.materials = this->material_params();
        cut_inp.particles = this->particle_params();
        cut_inp.cutoffs = {{pdg::electron(), {{MevEnergy{0.001}, 0.1234}}}};
        this->set_cutoff_params(cut_inp);

        // Set model data: default to ICRU73QO
        this->set_icru73qo();

        // Set default particle to muon with energy of 100 keV
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{0.1});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu");
    }

    void set_bragg()
    {
        auto const& particles = *this->particle_params();
        data_.inc_particle = particles.find(pdg::mu_plus());
        data_.electron = particles.find(pdg::electron());
        data_.electron_mass = particles.get(data_.electron).mass();
        data_.lowest_kin_energy = detail::bragg_lowest_kin_energy();
    }

    void set_icru73qo()
    {
        auto const& particles = *this->particle_params();
        data_.inc_particle = particles.find(pdg::mu_minus());
        data_.electron = particles.find(pdg::electron());
        data_.electron_mass = particles.get(data_.electron).mass();
        data_.lowest_kin_energy = detail::icru73qo_lowest_kin_energy();
    }

    void sanity_check(Interaction const& interaction) const
    {
        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(interaction.direction));
        EXPECT_EQ(Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());

        auto const& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(data_.electron, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    MuHadIonizationData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuHadIonizationTest, basic)
{
    std::vector<real_type> energy;
    std::vector<real_type> costheta;

    auto sample = [&](PDGNumber pdg) {
        // Reserve 4 secondaries, one for each sample
        int const num_samples = 4;
        this->resize_secondaries(num_samples);

        energy.clear();
        costheta.clear();

        this->set_inc_particle(pdg, MevEnergy{0.1});

        // Create the interactor
        MuHadIonizationInteractor interact(
            data_,
            this->particle_track(),
            this->cutoff_params()->get(MaterialId{0}),
            this->direction(),
            this->secondary_allocator());
        RandomEngine& rng = this->rng();

        // Produce four samples from the original incident energy
        for (int i : range(num_samples))
        {
            Interaction result = interact(rng);
            SCOPED_TRACE(result);
            this->sanity_check(result);

            EXPECT_EQ(result.secondaries.data(),
                      this->secondary_allocator().get().data() + i);

            energy.push_back(result.secondaries.front().energy.value());
            costheta.push_back(dot_product(
                result.direction, result.secondaries.front().direction));
        }

        EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

        // Next sample should fail because we're out of secondary buffer space
        {
            Interaction result = interact(rng);
            EXPECT_EQ(0, result.secondaries.size());
            EXPECT_EQ(Action::failed, result.action);
        }

        // No interaction when max secondary energy is below production cut
        {
            this->set_inc_particle(pdg, MevEnergy{0.0011});
            MuHadIonizationInteractor interact(
                data_,
                this->particle_track(),
                this->cutoff_params()->get(MaterialId{0}),
                this->direction(),
                this->secondary_allocator());

            Interaction result = interact(rng);
            EXPECT_EQ(0, result.secondaries.size());
            EXPECT_EQ(Action::unchanged, result.action);
        }
    };

    // Sample ICRU73QO model with incident mu-
    {
        this->set_icru73qo();
        sample(pdg::mu_minus());

        static double const expected_energy[] = {0.0014458653777536,
                                                 0.001251648293082,
                                                 0.0013192801865397,
                                                 0.00057619400045627};
        static double const expected_costheta[] = {0.86662579730412,
                                                   0.80560684873176,
                                                   0.82734134051617,
                                                   0.54491853032358};

        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    }
    // Sample Bragg model with incident mu+
    {
        this->set_bragg();
        sample(pdg::mu_plus());

        static double const expected_energy[] = {0.00022900204776481,
                                                 0.0014511488605566,
                                                 3.1983487781218e-05,
                                                 7.5949049601834e-05};
        static double const expected_costheta[] = {0.3429925946801,
                                                   0.86822876526498,
                                                   0.12806842199932,
                                                   0.19739055297736};
        EXPECT_VEC_SOFT_EQ(expected_energy, energy);
        EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    }
}

TEST_F(MuHadIonizationTest, stress_test)
{
    std::vector<real_type> avg_engine_samples;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_costheta;

    auto sample = [&](PDGNumber pdg) {
        unsigned int const num_samples = 10000;

        avg_engine_samples.clear();
        avg_energy.clear();
        avg_costheta.clear();

        for (real_type inc_e : {0.03, 0.05, 0.1, 0.15, 0.1999})
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            this->set_inc_particle(pdg, MevEnergy{inc_e});

            RandomEngine& rng = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;
            real_type energy = 0;
            real_type costheta = 0;

            // Loop over several incident directions
            for (Real3 const& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                // Create interactor
                MuHadIonizationInteractor interact(
                    data_,
                    this->particle_track(),
                    this->cutoff_params()->get(MaterialId{0}),
                    this->direction(),
                    this->secondary_allocator());

                // Loop over many particles
                for (unsigned int i = 0; i < num_samples; ++i)
                {
                    Interaction result = interact(rng);
                    this->sanity_check(result);

                    energy += result.secondaries.front().energy.value();
                    costheta += dot_product(
                        result.direction, result.secondaries.front().direction);
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(real_type(rng.count())
                                         / num_particles_sampled);
            avg_energy.push_back(energy / num_particles_sampled);
            avg_costheta.push_back(costheta / num_particles_sampled);
        }
    };

    // Sample ICRU73QO model with incident mu-
    {
        this->set_icru73qo();
        sample(pdg::mu_minus());

        static double const expected_avg_engine_samples[]
            = {6.0027, 6.0021, 6.003, 6.0034, 6.0047};
        static double const expected_avg_energy[] = {0.00056893310178363,
                                                     0.00072492000606412,
                                                     0.00097487866369081,
                                                     0.0011443155397256,
                                                     0.0012633301124393};
        static double const expected_avg_costheta[] = {0.99472414540371,
                                                       0.86537547828047,
                                                       0.69947394093299,
                                                       0.6113577317816,
                                                       0.55099275303428};
        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
    // Sample Bragg model with incident mu+
    {
        this->set_bragg();
        sample(pdg::mu_plus());

        static double const expected_avg_engine_samples[]
            = {6.0004, 6.0004, 6.0006, 6.0003, 6.0005};
        static double const expected_avg_energy[] = {8.8601911130921e-05,
                                                     0.00010210349779604,
                                                     0.00012023485252326,
                                                     0.00013103967324893,
                                                     0.00013806656798748};
        static double const expected_avg_costheta[] = {0.35858206068691,
                                                       0.29003901277676,
                                                       0.21408617527108,
                                                       0.17851819065736,
                                                       0.15626495143414};

        EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
        EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
        EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
