//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/BetheBloch.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/distribution/BetheBlochEnergyDistribution.hh"
#include "celeritas/em/interactor/MuHadIonizationInteractor.hh"
#include "celeritas/em/model/BetheBlochModel.hh"
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

class BetheBlochTest : public InteractorHostTestBase
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
        mu_minus.upper = MuIonizationProcess::Options{}.bethe_bloch_upper_limit;
        Applicability mu_plus = mu_minus;
        mu_plus.particle = particles.find(pdg::mu_plus());
        model_ = std::make_shared<BetheBlochModel>(
            ActionId{0}, particles, Model::SetApplicability{mu_minus, mu_plus});

        // Set default particle to muon with energy of 10 MeV
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{10});
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
    std::shared_ptr<BetheBlochModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(BetheBlochTest, distribution)
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

        BetheBlochEnergyDistribution sample(
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
        8268, 8482, 8377, 8202, 8302, 8416, 8240, 8429, 8371, 8267, 8338, 8308,
        8267, 8480, 8377, 8202, 8301, 8415, 8239, 8429, 8371, 8266, 8338, 8315,
        8267, 8480, 8377, 8202, 8301, 8415, 8239, 8429, 8370, 8266, 8338, 8316,
        8267, 8480, 8377, 8202, 8301, 8415, 8239, 8429, 8370, 8266, 8338, 8316,
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

TEST_F(BetheBlochTest, basic)
{
    // Reserve 4 secondaries, one for each sample
    int const num_samples = 4;
    this->resize_secondaries(num_samples);

    // Create the interactor
    MuHadIonizationInteractor<BetheBlochEnergyDistribution> interact(
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
    static double const expected_energy[] = {
        0.0071536254658967,
        0.0044460343975567,
        0.0051966857337682,
        0.0010332114718837,
    };
    static double const expected_costheta[] = {
        0.20412950131105,
        0.16112014959461,
        0.17413342534288,
        0.07778872399166,
    };

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
        MuHadIonizationInteractor<BetheBlochEnergyDistribution> interact(
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

TEST_F(BetheBlochTest, stress_test)
{
    unsigned int const num_samples = 10000;
    std::vector<double> avg_engine_samples;
    std::vector<double> avg_energy;
    std::vector<double> avg_costheta;

    for (real_type inc_e : {0.2, 1.0, 10.0, 1e2, 1e3, 1e4, 1e6, 1e8})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{inc_e});

        RandomEngine& rng = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;
        double energy = 0;
        double costheta = 0;

        // Loop over several incident directions
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            MuHadIonizationInteractor<BetheBlochEnergyDistribution> interact(
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

    // Looser tolerance for Windows build
    double const tol = 1e-11;

    // Gold values for average number of calls to RNG
    static double const expected_avg_engine_samples[]
        = {6.0069, 6.011, 6.0185, 6.0071, 6.0002, 6, 6, 6};
    static double const expected_avg_energy[] = {
        0.001820244315187,
        0.0030955371350616,
        0.0051011191515049,
        0.0071137840944271,
        0.010051462815079,
        0.0091324190520607,
        0.010625686784878,
        0.012563698826237,
    };
    static double const expected_avg_costheta[] = {
        0.67374005035636,
        0.37023194384465,
        0.14030216439644,
        0.06933001323056,
        0.060759849187212,
        0.060385882216971,
        0.060522416476511,
        0.060172252455851,
    };

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
    EXPECT_VEC_NEAR(expected_avg_energy, avg_energy, tol);
    EXPECT_VEC_SOFT_EQ(expected_avg_costheta, avg_costheta);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
