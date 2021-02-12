//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsStepUtils.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/PhysicsStepUtils.hh"

#include "base/PieStateStore.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "celeritas_test.hh"
#include "PhysicsTestBase.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PhysicsStepUtilsTest : public PhysicsTestBase
{
    using Base = PhysicsTestBase;

  protected:
    using ParticleStateStore = PieStateStore<ParticleStateData, MemSpace::host>;
    using PhysicsStateStore  = PieStateStore<PhysicsStateData, MemSpace::host>;

    void SetUp() override
    {
        Base::SetUp();

        // Construct state for a single host thread
        par_state  = ParticleStateStore(*this->particles(), 1);
        phys_state = PhysicsStateStore(*this->physics(), 1);
    }

    ParticleStateStore par_state;
    PhysicsStateStore  phys_state;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhysicsStepUtilsTest, calc_tabulated_physics_step)
{
    ParticleTrackView particle(
        this->particles()->host_pointers(), par_state.ref(), ThreadId{0});
    PhysicsTrackView                 phys(this->physics()->host_pointers(),
                          phys_state.ref(),
                          ParticleId{0},
                          MaterialId{0},
                          ThreadId{0});
    ParticleTrackView::Initializer_t par_init;
    PhysicsTrackView::Initializer_t  phys_init;

    // XXX add tests for a variety of energy ranges and multiple material IDs
    {
        par_init.energy      = MevEnergy{1e3};
        par_init.particle_id = particles()->find("celeriton");
        particle             = par_init;
        phys                 = phys_init;

        real_type step = celeritas::calc_tabulated_physics_step(particle, phys);
        EXPECT_EQ(0, step);
    }
}

TEST_F(PhysicsStepUtilsTest, calc_energy_loss) {}

TEST_F(PhysicsStepUtilsTest, select_model) {}
