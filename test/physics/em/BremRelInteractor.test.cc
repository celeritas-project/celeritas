//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremRelInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/BremRelInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialTrackView.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::BremRelInteractor;
using namespace celeritas;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class BremRelInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using constants::na_avogadro;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // XXX Update these based on particles needed by interactor
        Base::set_particle_params(
            {{"electron",
              pdg::electron(),
              MevMass{0.5109989461},
              ElementaryCharge{-1},
              stable},
             {"gamma", pdg::gamma(), zero, zero, stable}});
        const auto& params    = this->particle_params();
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());

        // Set default particle to incident XXX MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});

        // Create test materials
        Base::set_material_params({
            {
                {1, AmuMass{1.008}, "H"},
                {11, AmuMass{22.98976928}, "Na"},
                {53, AmuMass{126.90447}, "I"},
            },
            {
                {1e-5 * constants::na_avogadro,
                 100.0,
                 MatterState::gas,
                 {{ElementId{0}, 1.0}},
                 "H2"},
                {0.05 * constants::na_avogadro,
                 293.0,
                 MatterState::solid,
                 {{ElementId{1}, 0.5}, {ElementId{2}, 0.5}},
                 "NaI"},
            },
        });
        this->set_material("NaI");
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

        // XXX Check secondaries

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }

  protected:
    celeritas::BremRelInteractorPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(BremRelInteractorTest, basic)
{
    // Temporary test of harness material track view
    MaterialTrackView& mat_track = this->material_track();
    EXPECT_EQ(2, mat_track.element_scratch().size());
    EXPECT_EQ(MaterialId{1}, mat_track.def_id());
}
