//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/Stepper.hh"

#include <memory>
#include <random>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "DummyAction.hh"
#include "StepperTestBase.hh"
#include "celeritas_test.hh"
#include "../SimpleTestBase.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class SimpleComptonTest : public SimpleTestBase, public StepperTestBase
{
  public:
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        CELER_ASSERT(p.particle_id);
        p.energy = units::MevEnergy{100};
        p.track_id = TrackId{0};
        p.position = from_cm(Real3{-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }

    size_type max_average_steps() const override { return 100000; }
};

class StepperOrderTest : public SimpleComptonTest
{
  public:
    void SetUp()
    {
        auto& aux_reg = this->aux_reg();
        auto& action_reg = this->action_reg();

        dummy_params_ = std::make_shared<DummyParams>(aux_reg->next_id());
        aux_reg->insert(dummy_params_);

        // Note that order shouldn't matter; we deliberately add these out of
        // order.
        for (auto action_order : {ActionOrder::user_post,
                                  ActionOrder::user_start,
                                  ActionOrder::user_pre})
        {
            // Create a new action that can read data from the dummy params
            action_reg->insert(std::make_shared<DummyAction>(
                action_reg->next_id(),
                action_order,
                std::string("dummy-") + to_cstring(action_order),
                dummy_params_->aux_id()));
        }
    }

    template<MemSpace M>
    DummyState const& get_state(CoreState<M> const& core_state) const
    {
        return get<DummyState>(core_state.aux(), dummy_params_->aux_id());
    }

    std::shared_ptr<DummyParams> dummy_params_;
};

//---------------------------------------------------------------------------//
// Two boxes: compton with fake cross sections
//---------------------------------------------------------------------------//

TEST_F(SimpleComptonTest, setup)
{
    auto result = this->check_setup();
    static char const* expected_process[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_process, result.processes);
}

TEST_F(SimpleComptonTest, host)
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);

    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

TEST_F(SimpleComptonTest, TEST_IF_CELER_DEVICE(device))
{
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));
    auto result = this->run(step, num_primaries);
    if (this->is_default_build())
    {
        EXPECT_EQ(919, result.num_step_iters());
        EXPECT_SOFT_EQ(53.8125, result.calc_avg_steps_per_primary());
        EXPECT_EQ(RunResult::StepCount({1, 6}), result.calc_queue_hwm());
    }
    EXPECT_EQ(3, result.calc_emptying_step());
}

//---------------------------------------------------------------------------//

TEST_F(StepperOrderTest, setup)
{
    auto result = this->check_setup();
    static char const* const expected_processes[] = {"Compton scattering"};
    EXPECT_VEC_EQ(expected_processes, result.processes);
    static char const* const expected_actions[] = {
        "extend-from-primaries",
        "initialize-tracks",
        "dummy-user_start",
        "pre-step",
        "dummy-user_pre",
        "along-step-neutral",
        "physics-discrete-select",
        "scat-klein-nishina",
        "geo-boundary",
        "dummy-user_post",
        "extend-from-secondaries",
    };
    EXPECT_VEC_EQ(expected_actions, result.actions);
}

TEST_F(StepperOrderTest, host)
{
    constexpr auto M = MemSpace::host;
    size_type num_primaries = 32;
    size_type num_tracks = 64;

    Stepper<M> step(this->make_stepper_input(num_tracks));
    auto const& state
        = this->get_state(dynamic_cast<CoreState<M> const&>(step.state()));

    EXPECT_EQ(0, state.action_order.size());
    EXPECT_EQ(M, state.memspace);
    EXPECT_EQ(StreamId{0}, state.stream_id);
    EXPECT_EQ(num_tracks, state.size);

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));

    static char const* const expected_action_order[]
        = {"user_start", "user_pre", "user_post"};
    EXPECT_VEC_EQ(expected_action_order, state.action_order);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
