//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionSequence.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Types.hh"
#include "celeritas/user/ActionTimes.hh"

#include "ActionGroups.hh"
#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class StatusChecker;

//---------------------------------------------------------------------------//
/*!
 * Sequence of step actions to invoke as part of a single step.
 *
 * TODO accessors here are used by diagnostic output from celer-sim etc.;
 * perhaps make this public or add a diagnostic output for it?
 *
 * \note This class must be constructed \em after all actions have been added
 * to the action registry.
 */
class ActionSequence
{
  public:
    //!@{
    //! \name Type aliases
    using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
    using SPActionTimes = std::shared_ptr<ActionTimes>;
    using MapStrDbl = ActionTimes::MapStrDbl;
    //!@}

  public:
    //! Construction/execution options
    struct Options
    {
        SPActionTimes action_times;  //!< Call DeviceSynchronize and add timer
    };

  public:
    // Construct from an action registry and sequence options
    ActionSequence(ActionRegistry const&, Options options);

    //// INVOCATION ////

    // Call beginning-of-run actions.
    template<MemSpace M>
    void begin_run(CoreParams const& params, CoreState<M>& state);

    // Launch all actions with the given memory space.
    template<MemSpace M>
    void step(CoreParams const&, CoreState<M>& state);

    // Get the accumulated action times
    MapStrDbl get_action_times(AuxStateVec const&) const;

    //// ACCESSORS ////

    //! Get the ordered vector of actions in the sequence
    ActionGroupsT const& actions() const { return actions_; }

  private:
    ActionGroupsT actions_;
    Options options_;
    size_type num_actions_;
    std::shared_ptr<StatusChecker const> status_checker_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
