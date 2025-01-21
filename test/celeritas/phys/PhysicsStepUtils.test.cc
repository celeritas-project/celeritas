//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PhysicsStepUtils.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/phys/PhysicsStepUtils.hh"

#include "corecel/data/CollectionStateStore.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/MockTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"

#include "DiagnosticRngEngine.hh"
#include "MockProcess.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class PhysicsStepUtilsTest : public MockTestBase
{
    using Base = MockTestBase;
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

  protected:
    using MaterialStateStore
        = CollectionStateStore<MaterialStateData, MemSpace::host>;
    using ParticleStateStore
        = CollectionStateStore<ParticleStateData, MemSpace::host>;
    using PhysicsStateStore
        = CollectionStateStore<PhysicsStateData, MemSpace::host>;

    using MevEnergy = units::MevEnergy;

    PhysicsOptions build_physics_options() const override { return {}; }

    void SetUp() override
    {
        Base::SetUp();

        // Construct state for a single host thread
        mat_state = MaterialStateStore(this->material()->host_ref(), 1);
        par_state = ParticleStateStore(this->particle()->host_ref(), 1);
        phys_state = PhysicsStateStore(this->physics()->host_ref(), 1);
    }

    //! Random number generator
    RandomEngine& rng() { return rng_; }

    PhysicsTrackView init_track(MaterialTrackView* mat,
                                MaterialId mid,
                                ParticleTrackView* par,
                                char const* name,
                                MevEnergy energy)
    {
        CELER_EXPECT(mat && par);
        CELER_EXPECT(mid < this->material()->size());
        *mat = MaterialTrackView::Initializer_t{mid};

        ParticleTrackView::Initializer_t par_init;
        par_init.particle_id = this->particle()->find(name);
        CELER_EXPECT(par_init.particle_id);
        par_init.energy = energy;
        *par = par_init;

        PhysicsTrackView phys(this->physics()->host_ref(),
                              phys_state.ref(),
                              *par,
                              mat->material_id(),
                              TrackSlotId{0});
        return phys;
    }

    PhysicsStepView step_view()
    {
        return PhysicsStepView{
            this->physics()->host_ref(), phys_state.ref(), TrackSlotId{0}};
    }

    MaterialStateStore mat_state;
    ParticleStateStore par_state;
    PhysicsStateStore phys_state;
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(PhysicsStepUtilsTest, calc_physics_step_limit)
{
    MaterialTrackView material(
        this->material()->host_ref(), mat_state.ref(), TrackSlotId{0});
    ParticleTrackView particle(
        this->particle()->host_ref(), par_state.ref(), TrackSlotId{0});
    PhysicsStepView pstep = this->step_view();

    ActionId range_action;
    ActionId discrete_action;
    {
        auto const& scalars = this->physics()->host_ref().scalars;
        range_action = scalars.range_action();
        discrete_action = scalars.discrete_action();
        EXPECT_FALSE(scalars.fixed_step_action);
    }

    // Test a variety of energies and multiple material IDs
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        phys.interaction_mfp(1);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(discrete_action, step.action);
        EXPECT_SOFT_EQ(1. / 3.e-4, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{10});
        phys.interaction_mfp(1e-4);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(discrete_action, step.action);
        EXPECT_SOFT_EQ(1.e-4 / 9.e-3, to_cm(step.step));

        // Increase the distance to interaction so range limits the step length
        phys.interaction_mfp(1);
        step = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(0.48853333333333326, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{1e-2});
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(0.0016666666666666663, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(&material,
                                                 MaterialId{2},
                                                 &particle,
                                                 "anti-celeriton",
                                                 MevEnergy{1e-2});
        phys.interaction_mfp(1e-6);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(discrete_action, step.action);
        EXPECT_SOFT_EQ(1.e-6 / 9.e-1, to_cm(step.step));

        // Increase the distance to interaction so range limits the step length
        phys.interaction_mfp(1);
        step = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(1.4285714285714282e-5, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(&material,
                                                 MaterialId{2},
                                                 &particle,
                                                 "anti-celeriton",
                                                 MevEnergy{10});
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(0.014285714285714284, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{10});
        phys.interaction_mfp(1e-4);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(discrete_action, step.action);
        EXPECT_SOFT_EQ(1.e-4 / 9.e-3, to_cm(step.step));

        // Increase the distance to interaction so range limits the step length
        phys.interaction_mfp(1);
        step = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(0.48853333333333326, to_cm(step.step));
    }
    {
        // Test absurdly low energy (1 + E = 1)
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{1e-18});
        phys.interaction_mfp(1e-10);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(5.2704627669473019e-12, to_cm(step.step));
    }
    {
        // Celerino should have infinite step with no action
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "celerino", MevEnergy{1});
        phys.interaction_mfp(1.234);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(ActionId{}, step.action);
        EXPECT_SOFT_EQ(std::numeric_limits<real_type>::infinity(),
                       to_cm(step.step));
    }
}

TEST_F(PhysicsStepUtilsTest, calc_mean_energy_loss)
{
    MaterialTrackView material(
        this->material()->host_ref(), mat_state.ref(), TrackSlotId{0});
    ParticleTrackView particle(
        this->particle()->host_ref(), par_state.ref(), TrackSlotId{0});

    // input: cm; output: MeV
    auto calc_eloss = [&](PhysicsTrackView& phys, real_type step) -> real_type {
        // Calculate and store the energy loss range to PhysicsTrackView
        auto grid_id = phys.range_grid();
        auto calc_range = phys.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(particle.energy());
        phys.dedx_range(range);

        auto result
            = calc_mean_energy_loss(particle, phys, step * units::centimeter);
        return value_as<MevEnergy>(result);
    };

    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        if (CELERITAS_DEBUG)
        {
            // Can't calc eloss for photons
            EXPECT_THROW(calc_eloss(phys, 1e4), DebugError);
        }
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "celeriton", MevEnergy{10});
        real_type const eloss_rate = (0.2 + 0.4);  // MeV / cm

        // Tiny step: should still be linear loss (single process)
        EXPECT_SOFT_EQ(eloss_rate * 1e-6, calc_eloss(phys, 1e-6));

        // Long step (lose half energy) will call inverse lookup. The correct
        // answer (if range table construction was done over energy loss)
        // should be half since the slowing down rate is constant over all
        real_type step = 0.5 * particle.energy().value() / eloss_rate;
        EXPECT_SOFT_EQ(5, calc_eloss(phys, step));

        // Long step (lose half energy) will call inverse lookup. The correct
        // answer (if range table construction was done over energy loss)
        // should be half since the slowing down rate is constant over all
        step = 0.999 * particle.energy().value() / eloss_rate;
        EXPECT_SOFT_EQ(9.99, calc_eloss(phys, step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "electron", MevEnergy{1e-3});
        real_type const eloss_rate = 0.5;  // MeV / cm

        // Low energy particle which loses all its energy over the step will
        // call inverse lookup. Remaining range will be zero and eloss will be
        // equal to the pre-step energy.
        real_type step = value_as<MevEnergy>(particle.energy()) / eloss_rate;
        EXPECT_SOFT_EQ(1e-3, calc_eloss(phys, step));
    }
}

TEST_F(PhysicsStepUtilsTest,
       TEST_IF_CELERITAS_DOUBLE(select_discrete_interaction))
{
    MaterialTrackView material(
        this->material()->host_ref(), mat_state.ref(), TrackSlotId{0});
    ParticleTrackView particle(
        this->particle()->host_ref(), par_state.ref(), TrackSlotId{0});
    PhysicsStepView pstep = this->step_view();

    auto const model_offset
        = this->physics()->host_ref().scalars.model_to_action;

    // Test a variety of energy ranges and multiple material IDs
    {
        MaterialView mat_view(this->material()->host_ref(), MaterialId{0});
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        phys.interaction_mfp(1);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_SOFT_EQ(1. / 3.e-4, to_cm(step.step));

        // Testing cheat.
        PhysicsTrackView::PhysicsStateRef state_shortcut(phys_state.ref());
        state_shortcut.state[TrackSlotId{0}].interaction_mfp = 0;

        auto action = select_discrete_interaction(
            mat_view, particle, phys, pstep, this->rng());
        EXPECT_EQ(action.unchecked_get(), 0 + model_offset);

        action = select_discrete_interaction(
            mat_view, particle, phys, pstep, this->rng());
        EXPECT_EQ(action.unchecked_get(), 2 + model_offset);

        action = select_discrete_interaction(
            mat_view, particle, phys, pstep, this->rng());
        EXPECT_EQ(action.unchecked_get(), 2 + model_offset);
    }

    {
        MaterialView mat_view(this->material()->host_ref(), MaterialId{1});
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{10});
        phys.interaction_mfp(1);

        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_SOFT_EQ(0.48853333333333326, to_cm(step.step));

        // Testing cheat.
        PhysicsTrackView::PhysicsStateRef state_shortcut(phys_state.ref());
        state_shortcut.state[TrackSlotId{0}].interaction_mfp = 0;

        // The expected values are correlated with the values generated by the
        // random number generator and by the energy of particle.
        // The number of tries is picked so that each process is selected at
        // least once.
        std::vector<ActionId::size_type> models(13, ActionId::size_type(-1));
        for (auto i : range(models.size()))
        {
            auto action_id = select_discrete_interaction(
                mat_view, particle, phys, pstep, this->rng());
            models[i] = action_id.unchecked_get() - model_offset;
        }

        static ActionId::size_type const expected_models[]
            = {5, 8, 8, 8, 8, 8, 1, 5, 8, 8, 5, 5, 8};
        EXPECT_VEC_EQ(expected_models, models);
        EXPECT_EQ(56, this->rng().count());
    }

    {
        // Test the integral approach
        unsigned int num_samples = 10000;
        std::vector<real_type> inc_energy = {0.01, 0.01, 0.1, 10};
        std::vector<real_type> scaled_energy = {0.001, 0.00999, 0.001, 8};
        std::vector<real_type> acceptance_rate;

        auto reject_action
            = this->physics()->host_ref().scalars.integral_rejection_action();

        for (auto i : range(inc_energy.size()))
        {
            MaterialView mat_view(this->material()->host_ref(), MaterialId{0});
            PhysicsTrackView phys = this->init_track(&material,
                                                     MaterialId{0},
                                                     &particle,
                                                     "electron",
                                                     MevEnergy{inc_energy[i]});
            ParticleProcessId ppid{0};
            auto const& integral_process = phys.integral_xs_process(ppid);
            EXPECT_TRUE(integral_process);

            // Get the estimate of the maximum cross section over the step
            real_type xs_max = phys.calc_max_xs(
                integral_process, ppid, mat_view, particle.energy());
            pstep.per_process_xs(ppid) = xs_max;

            // Set the post-step energy
            particle.energy(MevEnergy{scaled_energy[i]});

            unsigned int count = 0;
            for (unsigned int j = 0; j < num_samples; ++j)
            {
                phys.reset_interaction_mfp();
                pstep.macro_xs(xs_max);

                auto action = select_discrete_interaction(
                    mat_view, particle, phys, pstep, this->rng());
                if (action != reject_action)
                    ++count;
            }
            acceptance_rate.push_back(real_type(count) / num_samples);
        }
        real_type const expected_acceptance_rate[]
            = {0.9204, 0.9999, 0.4972, 1};
        EXPECT_VEC_EQ(expected_acceptance_rate, acceptance_rate);
    }
}

//---------------------------------------------------------------------------//

class StepLimiterTest : public PhysicsStepUtilsTest
{
    PhysicsOptions build_physics_options() const override
    {
        PhysicsOptions opts;

        opts.light.min_range = inf;  // Use analytic range instead of scaled
        opts.fixed_step_limiter = 1e-3 * units::centimeter;
        return opts;
    }
};

TEST_F(StepLimiterTest, calc_physics_step_limit)
{
    MaterialTrackView material(
        this->material()->host_ref(), mat_state.ref(), TrackSlotId{0});
    ParticleTrackView particle(
        this->particle()->host_ref(), par_state.ref(), TrackSlotId{0});
    PhysicsStepView pstep = this->step_view();

    ActionId range_action;
    ActionId discrete_action;
    ActionId fixed_step_action;
    {
        auto const& scalars = this->physics()->host_ref().scalars;
        range_action = scalars.range_action();
        discrete_action = scalars.discrete_action();
        fixed_step_action = scalars.fixed_step_action;
    }

    {
        // Gammas should not be limited since they have no energy loss
        // processes
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        phys.interaction_mfp(1);
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(discrete_action, step.action);
        EXPECT_SOFT_EQ(1. / 3.e-4, to_cm(step.step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{1}, &particle, "celeriton", MevEnergy{1e-3});

        // Small energy: still range action
        StepLimit step
            = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(range_action, step.action);
        EXPECT_SOFT_EQ(0.00016666666666666663, to_cm(step.step));

        particle.energy(MevEnergy{1e-1});
        step = calc_physics_step_limit(material, particle, phys, pstep);
        EXPECT_EQ(fixed_step_action, step.action);
        EXPECT_SOFT_EQ(0.001, to_cm(step.step));
    }
}
//---------------------------------------------------------------------------//

class SplinePhysicsStepUtilsTest : public PhysicsStepUtilsTest
{
    PhysicsOptions build_physics_options() const override
    {
        PhysicsOptions opts;
        opts.spline_eloss_order = 2;
        return opts;
    }
};

TEST_F(SplinePhysicsStepUtilsTest, calc_mean_energy_loss)
{
    MaterialTrackView material(
        this->material()->host_ref(), mat_state.ref(), TrackSlotId{0});
    ParticleTrackView particle(
        this->particle()->host_ref(), par_state.ref(), TrackSlotId{0});

    // input: cm; output: MeV
    auto calc_eloss = [&](PhysicsTrackView& phys, real_type step) -> real_type {
        // Calculate and store the energy loss range to PhysicsTrackView
        auto grid_id = phys.range_grid();
        auto calc_range = phys.make_calculator<RangeCalculator>(grid_id);
        real_type range = calc_range(particle.energy());
        phys.dedx_range(range);

        auto result
            = calc_mean_energy_loss(particle, phys, step * units::centimeter);
        return value_as<MevEnergy>(result);
    };

    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "gamma", MevEnergy{1});
        if (CELERITAS_DEBUG)
        {
            // Can't calc eloss for photons
            EXPECT_THROW(calc_eloss(phys, 1e4), DebugError);
        }
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "celeriton", MevEnergy{10});
        real_type const eloss_rate = (0.2 + 0.4);  // MeV / cm

        // Tiny step: should still be linear loss (single process)
        EXPECT_SOFT_EQ(eloss_rate * 1e-6, calc_eloss(phys, 1e-6));

        // Long step (lose half energy) will call inverse lookup. The correct
        // answer (if range table construction was done over energy loss)
        // should be half since the slowing down rate is constant over all
        real_type step = 0.5 * particle.energy().value() / eloss_rate;
        EXPECT_SOFT_EQ(5, calc_eloss(phys, step));

        // Long step (lose half energy) will call inverse lookup. The correct
        // answer (if range table construction was done over energy loss)
        // should be half since the slowing down rate is constant over all
        step = 0.999 * particle.energy().value() / eloss_rate;
        EXPECT_SOFT_EQ(9.99, calc_eloss(phys, step));
    }
    {
        PhysicsTrackView phys = this->init_track(
            &material, MaterialId{0}, &particle, "electron", MevEnergy{1e-3});
        real_type const eloss_rate = 0.5;  // MeV / cm

        // Low energy particle which loses all its energy over the step will
        // call inverse lookup. Remaining range will be zero and eloss will be
        // equal to the pre-step energy.
        real_type step = value_as<MevEnergy>(particle.energy()) / eloss_rate;
        EXPECT_SOFT_EQ(1e-3, calc_eloss(phys, step));
    }
}
//---------------------------------------------------------------------------//

}  // namespace test
}  // namespace celeritas
