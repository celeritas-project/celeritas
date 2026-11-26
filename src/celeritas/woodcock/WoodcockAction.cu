//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockAction.cu
//---------------------------------------------------------------------------//
#include "WoodcockAction.hh"

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
#    include "detail/WoodcockExecutor.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch the Woodcock action on device.
 */
void WoodcockAction::step([[maybe_unused]] CoreParams const& params,
                          [[maybe_unused]] CoreStateDevice& state) const
{
#if CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE
    auto execute
        = make_along_step_track_executor(params.ptr<MemSpace::native>(),
                                         state.ptr(),
                                         this->action_id(),
                                         detail::WoodcockExecutor{});
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(*this, params, state, execute);
#else
    CELER_NOT_IMPLEMENTED("Woodcock tracking with ORANGE");
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
