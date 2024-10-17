//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionSequence.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "corecel/Types.hh"

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
 * \todo Refactor action times as "aux data" and as an end-gather action so
 * that this class can merge across states. Currently there's one sequence per
 * stepper which isn't right.
 */
class ActionSequence
{
  public:
    //!@{
    //! \name Type aliases
    using ActionGroupsT = ActionGroups<CoreParams, CoreState>;
    using VecDouble = std::vector<double>;
    //!@}

  public:
    //! Construction/execution options
    struct Options
    {
        bool action_times{false};  //!< Call DeviceSynchronize and add timer
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

    //// ACCESSORS ////

    //! Whether synchronization is taking place
    bool action_times() const { return options_.action_times; }

    //! Get the ordered vector of actions in the sequence
    ActionGroupsT const& actions() const { return actions_; }

    //! Get the corresponding accumulated time, if 'sync' or host called
    VecDouble const& accum_time() const { return accum_time_; }

  private:
    ActionGroupsT actions_;
    Options options_;
    VecDouble accum_time_;
    std::shared_ptr<StatusChecker const> status_checker_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
