//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KleinNishina.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/KleinNishinaInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::detail::KleinNishinaInteractor;
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class KleinNishinaInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});

        // TODO: this should be part of the process's data storage/management
        const auto& params    = this->particle_params();
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());
        pointers_.inv_electron_mass
            = 1 / (params.get(pointers_.electron_id).mass().value());

        // Set default particle to incident 10 MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);
        // SCOPED_TRACE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(interaction.direction));
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);

        // Check secondaries
        ASSERT_EQ(1, interaction.secondaries.size());
        const auto& electron = interaction.secondaries.front();
        EXPECT_TRUE(electron);
        EXPECT_EQ(pointers_.electron_id, electron.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  electron.energy.value());
        EXPECT_LT(0, electron.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(electron.direction));

        this->check_conservation(interaction);
    }

  protected:
    celeritas::detail::KleinNishinaPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(KleinNishinaInteractorTest, ten_mev)
{
    // Reserve 4 secondaries
    this->resize_secondaries(4);

    // Create the interactor
    KleinNishinaInteractor interact(pointers_,
                                    this->particle_track(),
                                    this->direction(),
                                    this->secondary_allocator());
    RandomEngine&          rng_engine = this->rng();

    std::vector<double> energy;
    std::vector<double> energy_electron;
    std::vector<double> costheta;
    std::vector<double> costheta_electron;

    // Produce four samples from the original incident energy/dir
    for (int i : celeritas::range(4))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);
        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        // Add actual results to vector
        energy.push_back(result.energy.value());
        costheta.push_back(
            celeritas::dot_product(result.direction, this->direction()));
        energy_electron.push_back(result.secondaries.front().energy.value());
        costheta_electron.push_back(celeritas::dot_product(
            result.secondaries.front().direction, this->direction()));
    }

    EXPECT_EQ(4, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    const double expected_energy[]
        = {0.4581502636229, 1.325852509857, 9.837250571445, 0.5250297816972};
    const double expected_costheta[] = {
        -0.0642523962721, 0.6656882878883, 0.9991545931877, 0.07782377978055};
    const double expected_energy_electron[]
        = {9.541849736377, 8.674147490143, 0.1627494285554, 9.474970218303};
    const double expected_costheta_electron[]
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
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(KleinNishinaInteractorTest, stress_test)
{
    RandomEngine& rng_engine = this->rng();

    const int           num_samples = 8192;
    std::vector<double> avg_engine_samples;

    for (double inc_e : {0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::gamma(), MevEnergy{inc_e});
        RandomEngine::size_type num_particles_sampled = 0;

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (const Real3& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(num_samples);

            // Create interactor
            KleinNishinaInteractor interact(pointers_,
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
        rng_engine.reset_count();
    }

    // PRINT_EXPECTED(avg_engine_samples);
    // Gold values for average number of calls to RNG
    const double expected_avg_engine_samples[]
        = {10.99816894531, 9.483154296875, 8.295532226562, 8.00439453125};
    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

TEST_F(KleinNishinaInteractorTest, distributions)
{
    RandomEngine& rng_engine = this->rng();

    const int    num_samples   = 10000;
    const double inc_energy    = 1;
    Real3        inc_direction = {0, 0, 1};
    this->set_inc_particle(pdg::gamma(), MevEnergy{inc_energy});
    this->set_inc_direction(inc_direction);
    this->resize_secondaries(num_samples);

    // Create interactor
    KleinNishinaInteractor interact(pointers_,
                                    this->particle_track(),
                                    this->direction(),
                                    this->secondary_allocator());

    int              nbins = 10;
    std::vector<int> eps_dist(nbins);
    std::vector<int> costheta_dist(nbins);

    // Loop over many particles
    for (int i = 0; i < num_samples; ++i)
    {
        Interaction out = interact(rng_engine);
        // Bin energy loss
        double eps     = out.energy.value() / inc_energy;
        int    eps_bin = eps * nbins;
        if (eps_bin >= 0 && eps_bin < nbins)
        {
            ++eps_dist[eps_bin];
        }

        // Bin directional change
        double costheta = celeritas::dot_product(inc_direction, out.direction);
        int ct_bin = (1 + costheta) / 2 * nbins; // Remap from [-1,1] to [0,1]
        if (ct_bin >= 0 && ct_bin < nbins)
        {
            ++costheta_dist[ct_bin];
        }
    }
    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());
    // PRINT_EXPECTED(eps_dist);
    // PRINT_EXPECTED(costheta_dist);
    const int expected_eps_dist[]
        = {0, 0, 2010, 1365, 1125, 1067, 1077, 1066, 1123, 1167};
    const int expected_costheta_dist[]
        = {495, 459, 512, 528, 565, 701, 803, 1101, 1693, 3143};
    EXPECT_VEC_EQ(expected_eps_dist, eps_dist);
    EXPECT_VEC_EQ(expected_costheta_dist, costheta_dist);
}
