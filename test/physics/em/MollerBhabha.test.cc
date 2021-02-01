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
        constexpr auto zero   = celeritas::zero_quantity();
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
             {{celeritas::ElementDefId{0}, 1.0}},
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
        EXPECT_EQ(pointers_.electron_id, electron.def_id);
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

    // Get the ElementView
    const celeritas::ElementView element(
        this->material_track().material_view().element_view(
            celeritas::ElementComponentId{0}));

    // Create interactor
    MollerBhabhaInteractor mb_interactor(pointers_,
                                         this->particle_track(),
                                         this->direction(),
                                         this->secondary_allocator(),
                                         element);

    RandomEngine& rng_engine = this->rng();
    Interaction   result     = mb_interactor(rng_engine);
    this->sanity_check(result);
}

//---------------------------------------------------------------------------//
TEST_F(MollerBhabhaInteractorTest, bhabha_scattering_10_MeV)
{
    this->resize_secondaries(1);

    // Set 10 MeV positron for testing Bhabha scattering
    this->set_inc_particle(pdg::positron(), MevEnergy{10});

    // Get the ElementView
    const celeritas::ElementView element(
        this->material_track().material_view().element_view(
            celeritas::ElementComponentId{0}));

    // Create interactor
    MollerBhabhaInteractor mb_interactor(pointers_,
                                         this->particle_track(),
                                         this->direction(),
                                         this->secondary_allocator(),
                                         element);

    RandomEngine& rng_engine = this->rng();
    Interaction   result     = mb_interactor(rng_engine);
    this->sanity_check(result);
}
