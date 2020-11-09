//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInteractor.test.cc
//---------------------------------------------------------------------------//
#include "physics/em/RayleighInteractor.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using celeritas::RayleighInteractor;
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RayleighInteractorTest : public celeritas_test::InteractorHostTestBase
{
    using Base = celeritas_test::InteractorHostTestBase;

  protected:
    void SetUp() override
    {
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        celeritas::ZeroQuantity zero;
        auto                    stable = ParticleDef::stable_decay_constant();

        // XXX Update these based on particles needed by interactor
        Base::set_particle_params(
            {{{"electron", pdg::electron()},
              {MevMass{0.5109989461}, ElementaryCharge{-1}, stable}},
             {{"gamma", pdg::gamma()}, {zero, zero, stable}}});
        const auto& params    = this->particle_params();
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());

        // Set default particle to incident XXX MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{10});
        this->set_inc_direction({0, 0, 1});
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
    celeritas::RayleighInteractorPointers pointers_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(RayleighInteractorTest, basic) {}
