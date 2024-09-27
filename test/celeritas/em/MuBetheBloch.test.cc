//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MuBetheBloch.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/MuBBEnergyDistribution.hh"
#include "celeritas/em/interactor/MuHadIonizationInteractor.hh"
#include "celeritas/em/model/MuBetheBlochModel.hh"
#include "celeritas/em/process/MuIonizationProcess.hh"
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

class MuBetheBlochTest : public InteractorHostTestBase
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

        // Set model data
        auto const& particles = *this->particle_params();
        Applicability mu_minus;
        mu_minus.particle = particles.find(pdg::mu_minus());
        mu_minus.lower
            = MuIonizationProcess::Options{}.bragg_icru73qo_upper_limit;
        mu_minus.upper = detail::high_energy_limit();
        Applicability mu_plus = mu_minus;
        mu_plus.particle = particles.find(pdg::mu_plus());
        model_ = std::make_shared<MuBetheBlochModel>(
            ActionId{0}, particles, Model::SetApplicability{mu_minus, mu_plus});

        // Set default particle to muon with energy of 1 GeV
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{1e3});
        this->set_inc_direction({0, 0, 1});
        this->set_material("Cu");
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
        EXPECT_EQ(model_->host_ref().electron, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(electron.direction));

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    std::shared_ptr<MuBetheBlochModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuBetheBlochTest, distribution)
{
    int num_samples = 100000;
    int num_bins = 12;

    MevEnergy cutoff{0.001};

    std::vector<int> counters;
    std::vector<real_type> min_energy;
    std::vector<real_type> max_energy;
    for (real_type energy : {0.2, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e7})
    {
        this->set_inc_particle(pdg::mu_minus(), MevEnergy(energy));
        std::mt19937 rng;

        MuBBEnergyDistribution sample(
            this->particle_track(), cutoff, model_->host_ref().electron_mass);
        real_type min = value_as<MevEnergy>(sample.min_secondary_energy());
        real_type max = value_as<MevEnergy>(sample.max_secondary_energy());

        std::vector<int> count(num_bins);
        for ([[maybe_unused]] int i : range(num_samples))
        {
            auto r = value_as<MevEnergy>(sample(rng));
            ASSERT_GE(r, min);
            ASSERT_LE(r, max);
            int bin = int((1 / r - 1 / min) / (1 / max - 1 / min) * num_bins);
            CELER_ASSERT(bin >= 0 && bin < num_bins);
            ++count[bin];
        }
        counters.insert(counters.end(), count.begin(), count.end());
        min_energy.push_back(min);
        max_energy.push_back(max);
    }

    static int const expected_counters[] = {
        8273, 8487, 8379, 8203, 8303, 8419, 8249, 8422, 8366, 8265, 8334, 8300,
        8281, 8499, 8383, 8211, 8309, 8428, 8256, 8435, 8371, 8263, 8320, 8244,
        8294, 8514, 8391, 8225, 8326, 8440, 8269, 8446, 8391, 8279, 8321, 8104,
        8283, 8499, 8389, 8211, 8312, 8427, 8257, 8440, 8380, 8278, 8340, 8184,
        8272, 8488, 8374, 8191, 8308, 8410, 8232, 8419, 8370, 8281, 8338, 8317,
        8278, 8490, 8370, 8191, 8311, 8404, 8246, 8413, 8371, 8255, 8335, 8336,
        8308, 8467, 8361, 8164, 8309, 8404, 8226, 8442, 8359, 8296, 8330, 8334,
        8353, 8428, 8388, 8264, 8315, 8312, 8227, 8499, 8323, 8280, 8266, 8345,
    };
    static double const expected_min_energy[]
        = {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};
    static double const expected_max_energy[] = {
        0.0038354680957569,
        0.019248476995285,
        0.20048052363148,
        2.7972680400033,
        100.69707462436,
        4855.7535710157,
        90256.629501068,
        9989193.9209199,
    };
    EXPECT_VEC_EQ(expected_counters, counters);
    EXPECT_VEC_SOFT_EQ(expected_min_energy, min_energy);
    EXPECT_VEC_SOFT_EQ(expected_max_energy, max_energy);
}

TEST_F(MuBetheBlochTest, basic)
{
    // Reserve 4 secondaries, one for each sample
    int const num_samples = 4;
    this->resize_secondaries(num_samples);

    // Create the interactor
    MuHadIonizationInteractor<MuBBEnergyDistribution> interact(
        model_->host_ref(),
        this->particle_track(),
        this->cutoff_params()->get(MaterialId{0}),
        this->direction(),
        this->secondary_allocator());
    RandomEngine& rng = this->rng();

    std::vector<real_type> energy;
    std::vector<real_type> costheta;

    // Produce four samples from the original incident energy
    for (int i : range(num_samples))
    {
        Interaction result = interact(rng);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        energy.push_back(result.secondaries.front().energy.value());
        costheta.push_back(dot_product(result.direction,
                                       result.secondaries.front().direction));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    static double const expected_energy[] = {0.0073808587493352,
                                             0.0045240316369054,
                                             0.0010035512057465,
                                             0.0010192538277565};
    static double const expected_costheta[] = {0.085027068970677,
                                               0.066660728134886,
                                               0.031450169056164,
                                               0.031695022051136};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }

    // No interaction when max secondary energy is below production cut
    {
        CutoffParams::Input cut_inp;
        cut_inp.materials = this->material_params();
        cut_inp.particles = this->particle_params();
        cut_inp.cutoffs = {{pdg::electron(), {{MevEnergy{0.01}, 0.1234}}}};
        this->set_cutoff_params(cut_inp);

        this->set_inc_particle(pdg::mu_minus(), MevEnergy{0.2});
        MuHadIonizationInteractor<MuBBEnergyDistribution> interact(
            model_->host_ref(),
            this->particle_track(),
            this->cutoff_params()->get(MaterialId{0}),
            this->direction(),
            this->secondary_allocator());

        Interaction result = interact(rng);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::unchanged, result.action);
    }
}

TEST_F(MuBetheBlochTest, stress_test)
{
    unsigned int const num_samples = 10000;
    std::vector<real_type> avg_engine_samples;
    std::vector<real_type> avg_energy;
    std::vector<real_type> avg_costheta;

    for (real_type inc_e : {0.2, 1.0, 10.0, 1e2, 1e3, 1e4, 1e6, 1e8})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{inc_e});

        RandomEngine& rng = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;
        real_type energy = 0;
        real_type costheta = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            MuHadIonizationInteractor<MuBBEnergyDistribution> interact(
                model_->host_ref(),
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
                costheta += dot_product(result.direction,
                                        result.secondaries.front().direction);
            }
            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(real_type(rng.count())
                                     / num_particles_sampled);
        avg_energy.push_back(energy / num_particles_sampled);
        avg_costheta.push_back(costheta / num_particles_sampled);
    }

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[]
        = {6.0069, 6.011, 6.0185, 6.0071, 6.043, 6.1304, 6.456, 6.9743};
    static double const expected_avg_energy[] = {0.001820244315187,
                                                 0.0030955371350616,
                                                 0.0051011191515049,
                                                 0.0071137840944271,
                                                 0.011366437776212,
                                                 0.012948850359578,
                                                 0.011869147598544,
                                                 0.037634371734214};
    static double const expected_avg_costheta[] = {0.67374005035636,
                                                   0.37023194384465,
                                                   0.14030216439644,
                                                   0.06933001323056,
                                                   0.060919687684128,
                                                   0.060365597604504,
                                                   0.061014987960578,
                                                   0.060456801678551};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_SOFT_EQ(expected_avg_energy, avg_energy);
    EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
