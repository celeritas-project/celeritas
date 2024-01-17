//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ActionSequence.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"

#include "../ActionInterface.hh"
#include "../CoreTrackDataFwd.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ActionRegistry;
class CoreParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Sequence of explicit actions to invoke as part of a single step.
 *
 * TODO accessors here are used by diagnostic output from celer-sim etc.;
 * perhaps make this public or add a diagnostic output for it?
 */
class ActionSequence
{
  public:
    //!@{
    //! \name Type aliases
    using SPBegin = std::shared_ptr<BeginRunActionInterface>;
    using SPConstExplicit = std::shared_ptr<ExplicitActionInterface const>;
    using VecBeginAction = std::vector<SPBegin>;
    using VecExplicitAction = std::vector<SPConstExplicit>;
    using VecDouble = std::vector<double>;
    //!@}

    //! Construction/execution options
    struct Options
    {
        bool sync{false};  //!< Call DeviceSynchronize and add timer
    };

  public:
    // Construct from an action registry and sequence options
    ActionSequence(ActionRegistry const&, Options options);

    //// INVOCATION ////

    // Launch all actions with the given memory space.
    template<MemSpace M>
    void begin_run(CoreParams const& params, CoreState<M>& state);

    // Launch all actions with the given memory space.
    template<MemSpace M>
    void execute(CoreParams const& params, CoreState<M>& state);

    //// ACCESSORS ////

    //! Whether synchronization is taking place
    bool sync() const { return options_.sync; }

    //! Get the set of beginning-of-run actions
    VecBeginAction const& begin_run_actions() const { return begin_run_; }

    //! Get the ordered vector of actions in the sequence
    VecExplicitAction const& actions() const { return actions_; }

    //! Get the corresponding accumulated time, if 'sync' or host called
    VecDouble const& accum_time() const { return accum_time_; }

  private:
    Options options_;
    VecBeginAction begin_run_;
    VecExplicitAction actions_;
    VecDouble accum_time_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
