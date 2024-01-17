//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include <string>

#include "corecel/Types.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "BoundaryExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
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
void BoundaryAction::execute(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              BoundaryExecutor{});
    return launch_action(*this, params, state, execute);
}

#if !CELER_USE_DEVICE
void BoundaryAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
