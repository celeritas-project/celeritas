//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerBhabha.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/detail/MollerBhabhaInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"
#include "physics/material/MaterialTrackView.hh"

using celeritas::Action;
using celeritas::detail::MollerBhabhaInteractor;
using celeritas::units::AmuMass;
namespace constants = celeritas::constants;
namespace pdg       = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class MollerBhabhaInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Particles needed by interactor
        Base::set_particle_params({{"electron",
                                    pdg::electron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{-1},
                                    stable},

                                   {"positron",
                                    pdg::positron(),
                                    MevMass{0.5109989461},
                                    ElementaryCharge{1},
                                    stable}});

        const auto& params           = this->particle_params();
        pointers_.electron_id        = params.find(pdg::electron());
        pointers_.positron_id        = params.find(pdg::positron());
        pointers_.electron_mass_c_sq = 0.5109989461;
        pointers_.min_valid_energy_  = MevEnergy{1e-3};

        // Set default incident direction. Particle is defined in the tests
        this->set_inc_direction({0, 0, 1});

        // Setup MaterialView
        MaterialParams::Input inp;
        inp.elements  = {{29, AmuMass{63.546}, "Cu"}};
        inp.materials = {
            {1.0 * constants::na_avogadro,
             293.0,
             celeritas::MatterState::solid,
             {{celeritas::ElementId{0}, 1.0}},
             "Cu"},
        };
        this->set_material_params(inp);
        this->set_material("Cu");
    }

    void sanity_check(const Interaction& interaction) const
    {
        ASSERT_TRUE(interaction);

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

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    celeritas::detail::MollerBhabhaPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(MollerBhabhaInteractorTest, moller_scattering_10_MeV)
{
    this->resize_secondaries(1);

    // Set 10 MeV electron for testing Moller scattering
    this->set_inc_particle(pdg::electron(), MevEnergy{10});

    // Create interactor
    MollerBhabhaInteractor mb_interactor(pointers_,
                                         this->particle_track(),
                                         this->direction(),
                                         this->secondary_allocator());

    RandomEngine& rng_engine = this->rng();
    Interaction   result     = mb_interactor(rng_engine);
    this->sanity_check(result);

    // Incident particle
    Real3 expected_inc_exiting_direction
        = {0.00110567321905880, -0.00288939665156339, 0.99999521442541039};

    EXPECT_EQ(Action::scattered, result.action);
    EXPECT_VEC_SOFT_EQ(expected_inc_exiting_direction, result.direction);
    EXPECT_SOFT_EQ(9.99896787404501630, result.energy.value());
    EXPECT_EQ(0.0, result.energy_deposition.value());
    EXPECT_EQ(1, result.secondaries.size());

    // Secondary
    Real3 expected_secondary_direction
        = {-0.35719359718530397, 0.93343491175959759, 0.03334665857586167};
    auto secondary = result.secondaries[0];

    EXPECT_GE(0, secondary.particle_id.get());
    EXPECT_VEC_SOFT_EQ(expected_secondary_direction, secondary.direction);
    EXPECT_SOFT_EQ(0.00103212595498286, secondary.energy.value());
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, bhabha_scattering_10_MeV)
{
    this->resize_secondaries(1);

    // Set 10 MeV positron for testing Bhabha scattering
    this->set_inc_particle(pdg::positron(), MevEnergy{10});

    // Create interactor
    MollerBhabhaInteractor mb_interactor(pointers_,
                                         this->particle_track(),
                                         this->direction(),
                                         this->secondary_allocator());

    RandomEngine& rng_engine = this->rng();
    Interaction   result     = mb_interactor(rng_engine);
    this->sanity_check(result);

    // Incident particle
    Real3 expected_inc_exiting_direction
        = {0.00110567499562512, -0.00288940129416909, 0.99999521441003170};

    EXPECT_EQ(Action::scattered, result.action);
    EXPECT_VEC_SOFT_EQ(expected_inc_exiting_direction, result.direction);
    EXPECT_SOFT_EQ(9.99896787072854032, result.energy.value());
    EXPECT_EQ(0.0, result.energy_deposition.value());
    EXPECT_EQ(1, result.secondaries.size());

    // Secondary
    Real3 expected_secondary_direction
        = {-0.35719359654708832, 0.93343491009178281, 0.03334671209731647};
    auto secondary = result.secondaries[0];

    EXPECT_GE(0, secondary.particle_id.get());
    EXPECT_VEC_SOFT_EQ(expected_secondary_direction, secondary.direction);
    EXPECT_SOFT_EQ(0.00103212927146000, secondary.energy.value());
}
