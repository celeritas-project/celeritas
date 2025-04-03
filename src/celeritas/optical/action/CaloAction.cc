//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/CaloAction.cc
//---------------------------------------------------------------------------//
#include "CaloAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/CaloExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
CaloAction::CaloAction(ActionId aid)
    : StaticConcreteAction(aid, "track-calo", "contribute to detector")
{
    CELER_LOG(status) << "Creating Calo action";
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void CaloAction::step(CoreParams const& params, CoreStateHost& state) const
{
    CELER_LOG(status) << "Performing CaloAction Step";
    auto detect_and_update
        = [](CoreTrackView& track) { detail::CaloExecutor{}(track); };
    auto execute = make_active_thread_executor(
        params.ptr<MemSpace::native>(), state.ptr(), detect_and_update);
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void CaloAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas