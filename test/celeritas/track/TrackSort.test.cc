
//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackSort.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <memory>
#include <vector>

#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/detail/TrackSortUtils.hh"

#include "../global/Stepper.test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

using celeritas::units::MevEnergy;

#define TestTrackPartitionEm3Stepper \
    TEST_IF_CELERITAS_GEANT(TestTrackPartitionEm3Stepper)
class TestTrackPartitionEm3Stepper : public TestEm3NoMsc
{
  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::partition_status;
        return std::make_shared<TrackInitParams>(input);
    }
};

#define TestTrackSortActionIdEm3Stepper \
    TEST_IF_CELERITAS_GEANT(TestTrackSortActionIdEm3Stepper)
class TestTrackSortActionIdEm3Stepper : public TestEm3NoMsc
{
  protected:
    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 4096;
        input.max_events = 4096;
        input.track_order = TrackOrder::sort_step_limit_action;
        return std::make_shared<TrackInitParams>(input);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TestTrackPartitionEm3Stepper, host_is_partitioned)
{
    size_type num_primaries = 8;
    size_type num_tracks = 128;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));
    auto check_is_partitioned = [&step] {
        auto span
            = step.state_ref().track_slots[AllItems<TrackSlotId::size_type>{}];
        return std::is_partitioned(
            span.begin(),
            span.end(),
            [&status = step.state_ref().sim.status](auto const track_slot) {
                return status[TrackSlotId{track_slot}] == TrackStatus::alive;
            });
    };
    // we partition at the start of the step so we need to explictly partition
    // again after a step before checking
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::partition_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::partition_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
}

TEST_F(TestTrackPartitionEm3Stepper,
       TEST_IF_CELER_DEVICE(device_is_partitioned))
{
    size_type num_primaries = 8;
    // Num tracks is low enough to hit capacity
    size_type num_tracks = num_primaries * 800;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));
    auto check_is_partitioned = [&step] {
        // copy to host
        auto const& state_ref = step.state_ref();
        Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>
            track_slots;
        track_slots = state_ref.track_slots;
        StateCollection<TrackStatus, Ownership::value, MemSpace::host> track_status;
        track_status = state_ref.sim.status;

        // check for partitioned tracks
        auto span = track_slots[AllItems<TrackSlotId::size_type>{}];
        return std::is_partitioned(
            span.begin(), span.end(), [&track_status](auto const track_slot) {
                return track_status[TrackSlotId{track_slot}]
                       == TrackStatus::alive;
            });
    };
    // we partition at the start of the step so we need to explictly partition
    // again after a step before checking
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::partition_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(), TrackOrder::partition_status);
        EXPECT_TRUE(check_is_partitioned()) << "Track slots are not "
                                               "partitioned by status";
        step();
    }
}

TEST_F(TestTrackSortActionIdEm3Stepper, host_is_sorted)
{
    size_type num_primaries = 8;
    size_type num_tracks = 128;

    Stepper<MemSpace::host> step(this->make_stepper_input(num_tracks));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));
    auto check_is_sorted = [&step] {
        auto& step_limit = step.state_ref().sim.step_limit;
        auto& track_slots = step.state_ref().track_slots;
        for (celeritas::size_type i = 0; i < track_slots.size() - 1; ++i)
        {
            TrackSlotId tid_current{track_slots[ThreadId{i}]},
                tid_next{track_slots[ThreadId{i + 1}]};
            ActionId::size_type aid_current{
                step_limit[tid_current].action.unchecked_get()},
                aid_next{step_limit[tid_next].action.unchecked_get()};
            EXPECT_LE(aid_current, aid_next)
                << aid_current << " is larger than " << aid_next;
        }
    };
    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::sort_step_limit_action);
        check_is_sorted();
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::sort_step_limit_action);
        check_is_sorted();
        step();
    }
}

TEST_F(TestTrackSortActionIdEm3Stepper, TEST_IF_CELER_DEVICE(device_is_sorted))
{
    size_type num_primaries = 8;
    // Num tracks is low enough to hit capacity
    size_type num_tracks = num_primaries * 800;

    Stepper<MemSpace::device> step(this->make_stepper_input(num_tracks));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(num_primaries);
    step(make_span(primaries));
    auto check_is_sorted = [&step] {
        // copy to host
        auto const& state_ref = step.state_ref();
        Collection<TrackSlotId::size_type, Ownership::value, MemSpace::host, ThreadId>
            track_slots;
        track_slots = state_ref.track_slots;
        StateCollection<StepLimit, Ownership::value, MemSpace::host> step_limit;
        step_limit = state_ref.sim.step_limit;

        for (celeritas::size_type i = 0; i < track_slots.size() - 1; ++i)
        {
            TrackSlotId tid_current{track_slots[ThreadId{i}]},
                tid_next{track_slots[ThreadId{i + 1}]};
            ActionId::size_type aid_current{
                step_limit[tid_current].action.unchecked_get()},
                aid_next{step_limit[tid_next].action.unchecked_get()};
            EXPECT_LE(aid_current, aid_next)
                << aid_current << " is larger than " << aid_next;
        }
    };
    // A step can change the step-limit action, so we need to redo the sorting
    // after taking a step.
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::sort_step_limit_action);
        check_is_sorted();
        step();
    }
    step(make_span(primaries));
    for (auto i = 0; i < 10; ++i)
    {
        detail::sort_tracks(step.state_ref(),
                            TrackOrder::sort_step_limit_action);
        check_is_sorted();
        step();
    }
}

}  // namespace test
}  // namespace celeritas
