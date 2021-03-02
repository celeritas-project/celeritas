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
    using MaterialStateStore = PieStateStore<MaterialStateData, MemSpace::host>;
    using ParticleStateStore = PieStateStore<ParticleStateData, MemSpace::host>;
    using PhysicsStateStore  = PieStateStore<PhysicsStateData, MemSpace::host>;

    PhysicsOptions build_physics_options() const override
    {
        PhysicsOptions opts;
        if (false)
        {
            // Don't scale the range -- use exactly the analytic values our
            // model has.
            opts.min_range = inf;
        }
        return opts;
    }

    void SetUp() override
    {
        Base::SetUp();

        // Construct state for a single host thread
        mat_state  = MaterialStateStore(*this->materials(), 1);
        par_state  = ParticleStateStore(*this->particles(), 1);
        phys_state = PhysicsStateStore(*this->physics(), 1);
    }

    PhysicsTrackView init_track(MaterialTrackView* mat,
                                MaterialId         mid,
                                ParticleTrackView* par,
                                const char*        name,
                                MevEnergy          energy)
    {
        CELER_EXPECT(mat && par);
        CELER_EXPECT(mid < this->materials()->size());
        *mat = MaterialTrackView::Initializer_t{mid};

        ParticleTrackView::Initializer_t par_init;
        par_init.particle_id = this->particles()->find(name);
        CELER_EXPECT(par_init.particle_id);
        par_init.energy = energy;
        *par            = par_init;

        PhysicsTrackView phys(this->physics()->host_pointers(),
                              phys_state.ref(),
                              par->particle_id(),
                              mat->material_id(),
                              ThreadId{0});
        return phys;
    }

    MaterialStateStore mat_state;
    ParticleStateStore par_state;
    PhysicsStateStore  phys_state;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhysicsStepUtilsTest, update_physics_step)
{
    MaterialTrackView material(
        this->materials()->host_pointers(), mat_state.ref(), ThreadId{0});
    ParticleTrackView particle(
        this->particles()->host_pointers(), par_state.ref(), ThreadId{0});

    // Test a variety of energies and multiple material IDs
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        celeritas::update_physics_step(material, particle, phys);
        EXPECT_SOFT_EQ(3.e-4, phys.macro_xs());
        EXPECT_SOFT_EQ(inf, phys.range_limit());
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{10});
        celeritas::update_physics_step(material, particle, phys);
        EXPECT_SOFT_EQ(9.e-3, phys.macro_xs());
        EXPECT_SOFT_EQ(0.6568, phys.range_limit());
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{1e-2});
        celeritas::update_physics_step(material, particle, phys);
        EXPECT_SOFT_EQ(0.009, phys.macro_xs());
        EXPECT_SOFT_EQ(0.0025, phys.range_limit());
    }
    {
        PhysicsTrackView phys = this->init_track(&material,
                                                 MaterialId{2},
                                                 &particle,
                                                 "anti-celeriton",
                                                 MevEnergy{1e-2});
        celeritas::update_physics_step(material, particle, phys);
        EXPECT_SOFT_EQ(9.e-1, phys.macro_xs());
        EXPECT_SOFT_EQ(2.5e-5, phys.range_limit());
    }
    {
        PhysicsTrackView phys = this->init_track(&material,
                                                 MaterialId{2},
                                                 &particle,
                                                 "anti-celeriton",
                                                 MevEnergy{10});
        celeritas::update_physics_step(material, particle, phys);
        EXPECT_SOFT_EQ(9.e-1, phys.macro_xs());
        EXPECT_SOFT_EQ(0.025, phys.range_limit());
    }
}

TEST_F(PhysicsStepUtilsTest, calc_energy_loss)
{
    MaterialTrackView material(
        this->materials()->host_pointers(), mat_state.ref(), ThreadId{0});
    ParticleTrackView particle(
        this->particles()->host_pointers(), par_state.ref(), ThreadId{0});

    // Test a variety of energies and multiple material IDs
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        phys.macro_xs(3e-4);
        phys.range_limit(inf);

        // Long step, but gamma means no energy loss
        EXPECT_SOFT_EQ(0, celeritas::calc_energy_loss(particle, phys, 1e4));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "celeriton", MevEnergy{10});
        phys.macro_xs(9e-4);
        phys.range_limit(25.0); // limit when min_range=inf

        // Tiny step: should still be linear loss (single process)
        const real_type eloss_rate = 0.2 + 0.4;
        EXPECT_SOFT_EQ(eloss_rate * 1e-6,
                       celeritas::calc_energy_loss(particle, phys, 1e-6));

        // Long step (lose half energy) will call inverse lookup. The correct
        // answer (if range table construction was done over energy loss)
        // should be half since the slowing down rate is constant over all
        const real_type step = 0.5 * particle.energy().value() / eloss_rate;
        EXPECT_SOFT_EQ(
            5, celeritas::calc_energy_loss(particle, phys, step).value());
    }
}

TEST_F(PhysicsStepUtilsTest, select_model) {}
