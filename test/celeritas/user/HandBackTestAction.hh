//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HandBackTestAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Hand back alive tracks whose post-step action matches a trigger.
 */
struct HandBackTestExecutor
{
    ActionId trigger;

    inline CELER_FUNCTION void operator()(CoreTrackView const& track) const;
};

//---------------------------------------------------------------------------//
/*!
 * Hand back tracks at the end of a step limited by the trigger action.
 *
 * This test action emulates a hand-back condition (e.g., leaving the region
 * transported on GPU) by marking every alive track whose post-step action is
 * the given trigger.
 */
class HandBackTestAction final : public CoreStepActionInterface
{
  public:
    // Construct with action ID and the post-step action that triggers it
    HandBackTestAction(ActionId id, ActionId trigger);

    // Run on host
    void step(CoreParams const&, CoreStateHost&) const final;
    // Run on device
    void step(CoreParams const&, CoreStateDevice&) const final;

    ActionId action_id() const final { return id_; }
    std::string_view label() const final { return "hand-back-test"; }
    std::string_view description() const final
    {
        return "mark tracks to be handed back for testing";
    }
    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    ActionId id_;
    ActionId trigger_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Hand back the track if it is alive and limited by the trigger action.
 */
CELER_FUNCTION void HandBackTestExecutor::operator()(
    CoreTrackView const& track) const
{
    auto sim = track.sim();
    if (sim.status() == TrackStatus::alive && sim.post_step_action() == trigger)
    {
        sim.hand_back(HandBackReason::user);
    }
}

#if !CELER_USE_DEVICE
inline void HandBackTestAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
