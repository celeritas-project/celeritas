//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DiscreteSelectAction.cc
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

#include <string>

#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "DiscreteSelectExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an action ID.
 */
DiscreteSelectAction::DiscreteSelectAction(ActionId aid)
    : ConcreteAction(
        aid, "physics-discrete-select", "select a discrete interaction")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on host.
 */
void DiscreteSelectAction::execute(CoreParams const& params,
                                   CoreStateHost& state) const
{
    auto execute = make_action_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              DiscreteSelectExecutor{});
    return launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
