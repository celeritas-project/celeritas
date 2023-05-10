//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
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
 */
class ActionSequence
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstExplicit = std::shared_ptr<ExplicitActionInterface const>;
    using VecAction = std::vector<SPConstExplicit>;
    using VecDouble = std::vector<double>;
    //!@}

    //! Construction/execution options
    struct Options
    {
        bool sync{false};  //!< Call DeviceSynchronize and add timer
    };

  public:
    // Empty sequence for delayed initialization
    ActionSequence() = default;

    // Construct from an action registry and sequence options
    ActionSequence(ActionRegistry const&, Options options);

    //// INVOCATION ////

    // Launch all actions with the given memory space.
    template<MemSpace M>
    void execute(CoreParams const& params, CoreState<M>& state);

    //// ACCESSORS ////

    //! Whether the sequence is assigned/valid
    explicit operator bool() const { return !actions_.empty(); }

    //! Whether synchronization is taking place
    bool sync() const { return options_.sync; }

    //! Get the ordered vector of actions in the sequence
    VecAction const& actions() const { return actions_; }

    //! Get the corresponding accumulated time, if 'sync' or host called
    VecDouble const& accum_time() const { return accum_time_; }

  private:
    Options options_;
    VecAction actions_;
    VecDouble accum_time_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
