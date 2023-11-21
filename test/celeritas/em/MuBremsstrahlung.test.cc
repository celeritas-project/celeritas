//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MuBremsstrahlung.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/MuBremsstrahlungInteractor.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/mat/MaterialView.hh"
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

class MuBremsstrahlungInteractorTest : public InteractorHostTestBase
{
    using Base = InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        auto const& params = this->particle_params();
        data_.ids.gamma = params->find(pdg::gamma());
        data_.ids.mu_minus = params->find(pdg::mu_minus());
        data_.ids.mu_plus = params->find(pdg::mu_plus());
        data_.electron_mass = params->get(params->find(pdg::electron())).mass();

        // Set default particle to muon with energy of 1100 MeV
        this->set_inc_particle(pdg::mu_minus(), MevEnergy{1100});
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

        auto const& gamma = interaction.secondaries.front();
        EXPECT_TRUE(gamma);
        EXPECT_EQ(data_.ids.gamma, gamma.particle_id);
        EXPECT_GT(this->particle_track().energy().value(),
                  gamma.energy.value());
        EXPECT_LT(0, gamma.energy.value());
        EXPECT_SOFT_EQ(1.0, norm(gamma.direction));

        // Check conservation between primary and secondaries
        // To be determined: Not sure if momentum is conserved.
        // this->check_conservation(interaction);
        this->check_energy_conservation(interaction);
    }

  protected:
    MuBremsstrahlungData data_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MuBremsstrahlungInteractorTest, basic)
{
    // Reserve 4 secondaries
    int num_samples = 4;
    this->resize_secondaries(num_samples);

    auto material = this->material_track().make_material_view();

    // Create the interactor
    MuBremsstrahlungInteractor interact(data_,
                                        this->particle_track(),
                                        this->direction(),
                                        this->secondary_allocator(),
                                        material,
                                        ElementComponentId{0});
    RandomEngine& rng_engine = this->rng();

    std::vector<double> energy;
    std::vector<double> costheta;

    // Produce four samples from the original incident energy
    for (int i : range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data() + i);

        energy.push_back(result.secondaries[0].energy.value());
        costheta.push_back(dot_product(result.secondaries.front().direction,
                                       this->direction()));
    }

    EXPECT_EQ(num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    double const expected_energy[] = {
        1012.99606184083, 1029.80705246907, 1010.52595539471, 1010.77666768483};
    double const expected_costheta[] = {0.968418002240112,
                                        0.999212413725981,
                                        0.998550042495312,
                                        0.983614606590488};

    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_costheta, costheta);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(Action::failed, result.action);
    }
}

TEST_F(MuBremsstrahlungInteractorTest, stress_test)
{
    unsigned int const num_samples = 1e4;
    std::vector<double> avg_engine_samples;

    for (auto particle : {pdg::mu_minus(), pdg::mu_plus()})
    {
        for (real_type inc_e : {1.5e4, 5e4, 10e4, 50e4, 100e4})
        {
            SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
            this->set_inc_particle(particle, MevEnergy{inc_e});

            RandomEngine& rng_engine = this->rng();
            RandomEngine::size_type num_particles_sampled = 0;

            // Loop over several incident directions
            for (Real3 const& inc_dir : {Real3{0, 0, 1},
                                         Real3{1, 0, 0},
                                         Real3{1e-9, 0, 1},
                                         Real3{1, 1, 1}})
            {
                SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
                this->set_inc_direction(inc_dir);
                this->resize_secondaries(num_samples);

                auto material = this->material_track().make_material_view();

                // Create interactor
                MuBremsstrahlungInteractor interact(data_,
                                                    this->particle_track(),
                                                    this->direction(),
                                                    this->secondary_allocator(),
                                                    material,
                                                    ElementComponentId{0});

                for (unsigned int i = 0; i < num_samples; i++)
                {
                    Interaction result = interact(rng_engine);
                    // SCOPED_TRACE(result);
                    this->sanity_check(result);
                }
                EXPECT_EQ(num_samples,
                          this->secondary_allocator().get().size());
                num_particles_sampled += num_samples;
            }
            avg_engine_samples.push_back(double(rng_engine.count())
                                         / double(num_particles_sampled));
        }
    }

    // Gold values for average number of calls to RNG
    double const expected_avg_engine_samples[] = {10.4316,
                                                  9.7148,
                                                  9.2378,
                                                  8.6495,
                                                  8.5108,
                                                  10.4121,
                                                  9.7221,
                                                  9.2726,
                                                  8.6439,
                                                  8.5178};

    EXPECT_VEC_SOFT_EQ(expected_avg_engine_samples, avg_engine_samples);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
