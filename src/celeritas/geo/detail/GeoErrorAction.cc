//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/detail/GeoErrorAction.cc
//---------------------------------------------------------------------------//
#include "GeoErrorAction.hh"

#include <string>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "GeoErrorExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
GeoErrorAction::GeoErrorAction(ActionId aid)
    : ConcreteAction(aid, "geo-kill", "kill a track due to a navigation error")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void GeoErrorAction::execute(CoreParams const& params,
                             CoreStateHost& state) const
{
    auto execute = make_action_track_executor(params.ptr<MemSpace::native>(),
                                              state.ptr(),
                                              this->action_id(),
                                              GeoErrorExecutor{});
    return launch_action(*this, params, state, execute);
}

#if !CELER_USE_DEVICE
void GeoErrorAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
