//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Physics.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/PhysicsTrackView.hh"
#include "physics/base/PhysicsStepUtils.hh"
#include "physics/base/PhysicsParams.hh"

#include "celeritas_test.hh"
#include "MockProcess.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/base/ParticleParams.hh"
// #include "Physics.test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS BASE
//---------------------------------------------------------------------------//

class PhysicsTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        using namespace celeritas::units;

        // Create materials
        {
            MaterialParams::Input inp;
            inp.elements = {{1, AmuMass{1.0}, "celerogen"},
                            {4, AmuMass{10.0}, "celerinium"}};
            inp.materials.push_back({1e20,
                                     300,
                                     MatterState::gas,
                                     {{ElementId{0}, 1.0}},
                                     "lo density celerogen"});
            inp.materials.push_back({1e21,
                                     300,
                                     MatterState::liquid,
                                     {{ElementId{0}, 1.0}},
                                     "hi density celerogen"});
            inp.materials.push_back({1e23,
                                     300,
                                     MatterState::solid,
                                     {{ElementId{1}, 1.0}},
                                     "solid celerinium"});
            materials = std::make_shared<MaterialParams>(std::move(inp));
        }
        // Create particles
        {
            namespace pdg = celeritas::pdg;

            constexpr auto zero = celeritas::zero_quantity();
            constexpr auto stable
                = celeritas::ParticleDef::stable_decay_constant();

            ParticleParams::Input inp;
            inp.push_back({"gamma", pdg::gamma(), zero, zero, stable});
            inp.push_back({"celeriton",
                           PDGNumber{1337},
                           MevMass{1},
                           ElementaryCharge{1},
                           stable});
            inp.push_back({"anti-celeriton",
                           PDGNumber{-1337},
                           MevMass{1},
                           ElementaryCharge{-1},
                           stable});
            particles = std::make_shared<ParticleParams>(std::move(inp));
        }
    }

    std::shared_ptr<MaterialParams> materials;
    std::shared_ptr<ParticleParams> particles;
    std::shared_ptr<PhysicsParams>  physics;
};

TEST_F(PhysicsTest, accessors)
{
    // PTestInput input;
    // input.num_threads = 0;
    // auto result = p_test(input);
    // PRINT_EXPECTED(result.foo);
    // EXPECT_VEC_SOFT_EQ(expected_foo, result.foo);
}

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class PhysicsTestHost : public PhysicsTest
{
  protected:
    void SetUp() override { CELER_NOT_IMPLEMENTED("physics step tests"); }

    struct
    {
        ParticleParamsPointers particle;
        PhysicsParamsPointers  physics;
    } params;
    struct
    {
        ParticleStatePointers particle;
        PhysicsStatePointers  physics;
    } states;
};

TEST_F(PhysicsTestHost, calc_tabulated_physics_step)
{
    ParticleTrackView particle(params.particle, states.particle, ThreadId{0});
    PhysicsTrackView  physics(params.physics,
                             states.physics,
                             ParticleId{0},
                             MaterialId{0},
                             ThreadId{0});

    // XXX add tests for a variety of energy ranges and multiple material IDs
    {
        states.particle.vars[0].energy = MevEnergy{1e3};
        real_type step
            = celeritas::calc_tabulated_physics_step(particle, physics);
        EXPECT_EQ(0, step);
    }
}

TEST_F(PhysicsTestHost, calc_energy_loss) {}

TEST_F(PhysicsTestHost, select_model) {}
