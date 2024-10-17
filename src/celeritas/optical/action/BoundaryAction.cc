//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/BoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
BoundaryAction::BoundaryAction(ActionId aid)
    : ConcreteAction(aid, "geo-boundary", "cross a geometry boundary")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void BoundaryAction::step(CoreParams const& params, CoreStateHost& state) const
{
    auto execute = make_action_thread_executor(params.ptr<MemSpace::native>(),
                                               state.ptr(),
                                               this->action_id(),
                                               detail::BoundaryExecutor{});
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void BoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
