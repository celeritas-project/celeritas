//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/EPlusGGInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::EPlusGGInteractor;
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class EPlusGGInteractorTest : public celeritas_test::InteractorHostTestBase
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
            {{{"electron", pdg::electron()},
              {MevMass{0.5109989461}, ElementaryCharge{-1}, stable}},
             {{"positron", pdg::positron()},
              {MevMass{0.5109989461}, ElementaryCharge{1}, stable}},
             {{"gamma", pdg::gamma()}, {zero, zero, stable}}});

        const auto& params      = this->particle_params();
        pointers_.positron_id   = params.find(pdg::positron());
        pointers_.gamma_id      = params.find(pdg::gamma());
        pointers_.electron_mass = 0.5109989461;

        // Set default particle to incident 10 MeV positron
        this->set_inc_particle(pdg::positron(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_EQ(0, interaction.energy.value());
        EXPECT_SOFT_EQ(0, celeritas::norm(interaction.direction));
        EXPECT_EQ(celeritas::Action::absorbed, interaction.action);

        // Check secondaries (two photons)
        ASSERT_EQ(2, interaction.secondaries.size());

        const auto& gamma1 = interaction.secondaries.front();
        EXPECT_TRUE(gamma1);
        EXPECT_EQ(pointers_.gamma_id, gamma1.def_id);

        EXPECT_GT(this->particle_track().energy().value()
                      + 2 * pointers_.electron_mass,
                  gamma1.energy.value());
        EXPECT_LT(0, gamma1.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(gamma1.direction));

        const auto& gamma2 = interaction.secondaries.back();
        EXPECT_TRUE(gamma2);
        EXPECT_EQ(pointers_.gamma_id, gamma2.def_id);
        EXPECT_GT(this->particle_track().energy().value()
                      + 2 * pointers_.electron_mass,
                  gamma2.energy.value());
        EXPECT_LT(0, gamma2.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(gamma2.direction));
    }

  protected:
    celeritas::EPlusGGInteractorPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EPlusGGInteractorTest, basic)
{
    const int num_samples = 4;

    // Reserve  num_samples*2 secondaries;
    this->resize_secondaries(num_samples * 2);

    // Create the interactor
    EPlusGGInteractor interact(pointers_,
                               this->particle_track(),
                               this->direction(),
                               this->secondary_allocator());
    RandomEngine&     rng_engine = this->rng();

    // Produce four samples from the original incident angle/energy
    std::vector<double> angle;
    std::vector<double> energy1;
    std::vector<double> energy2;

    for (int i : celeritas::range(num_samples))
    {
        Interaction result = interact(rng_engine);
        SCOPED_TRACE(result);
        this->sanity_check(result);

        EXPECT_EQ(result.secondaries.data(),
                  this->secondary_allocator().get().data()
                      + result.secondaries.size() * i);

        angle.push_back(
            celeritas::dot_product(result.secondaries.front().direction,
                                   result.secondaries.back().direction));
        energy1.push_back(result.secondaries[0].energy.value());
        energy2.push_back(result.secondaries[1].energy.value());
    }

    EXPECT_EQ(2 * num_samples, this->secondary_allocator().get().size());

    // Note: these are "gold" values based on the host RNG.
    const double expected_energy1[] = {
        9.58465334147939, 10.4793460046007, 3.88444170212412, 2.82099830657521};

    const double expected_energy2[] = {
        1.43734455072061, 0.542651887599318, 7.13755619007588, 8.20099958562479};

    const double expected_angle[] = {0.993884655017147,
                                     0.998663395567878,
                                     0.911748167069523,
                                     0.859684696937321};

    EXPECT_VEC_SOFT_EQ(expected_energy1, energy1);
    EXPECT_VEC_SOFT_EQ(expected_energy2, energy2);
    EXPECT_VEC_SOFT_EQ(expected_angle, angle);

    // Next sample should fail because we're out of secondary buffer space
    {
        Interaction result = interact(rng_engine);
        EXPECT_EQ(0, result.secondaries.size());
        EXPECT_EQ(celeritas::Action::failed, result.action);
    }
}

TEST_F(EPlusGGInteractorTest, stress_test)
{
    RandomEngine& rng_engine = this->rng();

    const int num_samples = 8192;

    for (double inc_e : {0.01, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE("Incident energy: " + std::to_string(inc_e));
        this->set_inc_particle(pdg::positron(), MevEnergy{inc_e});

        // Loop over several incident directions (shouldn't affect anything
        // substantial, but scattering near Z axis loses precision)
        for (const Real3& inc_dir :
             {Real3{0, 0, 1}, Real3{1, 0, 0}, Real3{1e-9, 0, 1}, Real3{1, 1, 1}})
        {
            SCOPED_TRACE("Incident direction: " + to_string(inc_dir));
            this->set_inc_direction(inc_dir);
            this->resize_secondaries(2 * num_samples);

            // Create interactor
            EPlusGGInteractor interact(pointers_,
                                       this->particle_track(),
                                       this->direction(),
                                       this->secondary_allocator());

            // Loop over many particles
            for (int i = 0; i < num_samples; ++i)
            {
                Interaction result = interact(rng_engine);
                SCOPED_TRACE(result);
                this->sanity_check(result);
            }
            EXPECT_EQ(2 * num_samples,
                      this->secondary_allocator().get().size());
        }
    }
}
