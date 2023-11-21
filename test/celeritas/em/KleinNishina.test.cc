//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/KleinNishina.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
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

class KleinNishinaInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        auto const& params = *this->particle_params();
        data_.ids.electron = params.find(pdg::electron());
        data_.ids.gamma = params.find(pdg::gamma());
        data_.inv_electron_mass
            = 1 / (params.get(data_.ids.electron).mass().value());

        // Set default particle to incident 10 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});
    }

    void sanity_check(Interaction const& interaction) const
    {
        // SCOPED_TRACE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(interaction.direction));
        EXPECT_EQ(Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());
        auto const& electron = interaction.secondaries.front();
        if (electron)
        {
            // Secondary survived cutoff
            EXPECT_EQ(data_.ids.electron, electron.particle_id);
            EXPECT_GT(this->particle_track().energy().value(),
                      electron.energy.value());
            EXPECT_LT(KleinNishinaInteractor::secondary_cutoff(),
                      electron.energy);
            EXPECT_EQ(0, interaction.energy_deposition.value());
            EXPECT_SOFT_EQ(1.0, norm(electron.direction));
        }
        else
        {
            // Secondary energy deposited locally
            EXPECT_GT(KleinNishinaInteractor::secondary_cutoff(),
                      interaction.energy_deposition);
            EXPECT_LT(0, interaction.energy_deposition.value());
        }

        // Since secondary cutoffs are applied inside the interactor, momentum
        // may not be conserved between the incoming and outgoing particles
        this->check_energy_conservation(interaction);
    }

  protected:
    KleinNishinaData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(KleinNishinaInteractorTest, ten_mev)
{
    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Create the interactor
    KleinNishinaInteractor interact(data_,
                                    this->particle_track(),
                                    this->direction(),
                                    this->secondary_allocator());
    RandomEngine& rng_engine = this->rng();

    std::vector<double> energy;
    std::vector<double> energy_electron;
    std::vector<double> costheta;
    std::vector<double> costheta_electron;

    // Produce four samples from the original incident energy/dir
    for (int i : range(4))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        // Add actual results to vector
        energy.push_back(result.energy.value());
        costheta.push_back(dot_product(result.direction, this->direction()));
        energy_electron.push_back(result.secondaries.front().energy.value());
        costheta_electron.push_back(dot_product(
            result.secondaries.front().direction, this->direction()));
    }

    EXPECT_EQ(4, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    double const expected_energy[]
        = {0.4581502636229, 1.325852509857, 9.837250571445, 0.5250297816972};
    double const expected_costheta[] = {
        -0.0642523962721, 0.6656882878883, 0.9991545931877, 0.07782377978055};
    double const expected_energy_electron[]
        = {9.541849736377, 8.674147490143, 0.1627494285554, 9.474970218303};
    double const expected_costheta_electron[]
        = {0.998962567429, 0.9941635460938, 0.3895748042313, 0.9986216572142};
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);
    EXPECT_VEC_SOFT_EQ(expected_energy_electron, energy_electron);
    EXPECT_VEC_SOFT_EQ(expected_costheta_electron, costheta_electron);
    // PRINT_EXPECTED(energy_electron);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(KleinNishinaInteractorTest, stress_test)
{
    int const num_samples = 8192;
    std::vector<double> avg_engine_samples;

    for (real_type inc_e : {0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});

        RandomEngine& rng_engine = this->rng();
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (Real3 const& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            KleinNishinaInteractor interact(data_,
                                            this->particle_track(),
                                            this->direction(),
                                            this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                // SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
            num_particles_sampled += num_samples;
        }
        avg_engine_samples.push_back(double(rng_engine.count())
                                     / double(num_particles_sampled));
    }

    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to RNG
    double const expected_avg_engine_samples[]
        = {10.99816894531, 9.483154296875, 8.295532226562, 8.00439453125};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(KleinNishinaInteractorTest, distributions)
{
    RandomEngine& rng_engine = this->rng();

    int const num_samples = 10000;
    real_type const inc_energy = 1;
    Real3 inc_direction = {0, 0, 1};
    this->set_inc_particle(pdg::gamma(), MevEnergy{inc_energy});
    this->set_inc_direction(inc_direction);
    this->resize_secondaries(num_samples);

    // Create interactor
    KleinNishinaInteractor interact(data_,
                                    this->particle_track(),
                                    this->direction(),
                                    this->secondary_allocator());

    int nbins = 10;
    std::vector<int> eps_dist(nbins);
    std::vector<int> costheta_dist(nbins);

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        // Bin energy loss
        double eps = out.energy.value() / inc_energy;
        int eps_bin = eps * nbins;
        if (eps_bin >= 0 && eps_bin < nbins)
        {
            ++eps_dist[eps_bin];
        }

        // Bin directional change
        double costheta = dot_product(inc_direction, out.direction);
        int ct_bin = (1 + costheta) / 2 * nbins;  // Remap from [-1,1] to [0,1]
        if (ct_bin >= 0 && ct_bin < nbins)
        {
            ++costheta_dist[ct_bin];
        }
    }
    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
    // PRINT_EXPECTED(eps_dist);
    // PRINT_EXPECTED(costheta_dist);
    int const expected_eps_dist[]
        = {0, 0, 2010, 1365, 1125, 1067, 1077, 1066, 1123, 1167};
    int const expected_costheta_dist[]
        = {495, 459, 512, 528, 565, 701, 803, 1101, 1693, 3143};
    EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
