//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockAction.cc
//---------------------------------------------------------------------------//
#include "WoodcockAction.hh"

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
#    include "detail/WoodcockExecutor.hh"  // IWYU pragma: associated
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with ID.
 */
WoodcockAction::WoodcockAction(ActionId aid)
    : StaticConcreteAction(
          aid, "woodcock", "move photons using a majorant cross section")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the Woodcock action on host.
 */
void WoodcockAction::step([[maybe_unused]] CoreParams const& params,
                          [[maybe_unused]] CoreStateHost& state) const
{
#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
    auto execute
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         this->action_id(),
                                         detail::WoodcockExecutor{});
    return launch_action(*this, params, state, execute);
#else
    CELER_NOT_IMPLEMENTED("Woodcock tracking with ORANGE");
#endif
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void WoodcockAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Construct with ID.
 */
WoodcockNullAction::WoodcockNullAction(ActionId aid)
    : StaticConcreteAction(
          aid, "woodcock-null", "pause after rejected Woodcock collisions")
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
