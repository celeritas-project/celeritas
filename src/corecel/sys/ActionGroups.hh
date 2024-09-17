//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionGroups.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;

//---------------------------------------------------------------------------//
/*!
 * Group and sequence actions to be used over the lifetime of a run.
 *
 * This class improves runtime performance by caching the correct base classes
 * of the virtual action hierarchy. It sorts step actions by increasing \c
 * StepActionOrder, then by increasing action ID. Other actions are sorted
 * solely by action ID.
 *
 * Because actions can inherit from multiple action types, the sum of actions
 * from these partitions may be \em greater than the number of actions in the
 * registry.
 */
template<class P, template<MemSpace M> class S>
class ActionGroups
{
  public:
    //!@{
    //! \name Type aliases
    using BeginRunActionT = BeginRunActionInterface<P, S>;
    using StepActionT = StepActionInterface<P, S>;
    using SPBeginAction = std::shared_ptr<BeginRunActionT>;
    using SPConstStepAction = std::shared_ptr<StepActionT const>;
    using VecBeginAction = std::vector<SPBeginAction>;
    using VecStepAction = std::vector<SPConstStepAction>;
    //!@}

  public:
    // Construct from an action registry and sequence options
    explicit ActionGroups(ActionRegistry const&);

    //! Get the set of beginning-of-run actions
    VecBeginAction const& begin_run() const { return begin_run_; }

    //! Get the ordered vector of actions within a step
    VecStepAction const& step() const { return step_actions_; }

  private:
    VecBeginAction begin_run_;
    VecStepAction step_actions_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
