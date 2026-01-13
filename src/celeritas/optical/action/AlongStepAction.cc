//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/AlongStepAction.cc
//---------------------------------------------------------------------------//
#include "AlongStepAction.hh"

#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

#include "ActionLauncher.hh"
#include "TrackSlotExecutor.hh"

#include "detail/AlongStepExecutor.hh"
#include "detail/PropagateExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
AlongStepAction::AlongStepAction(ActionId aid)
    : StaticConcreteAction(aid, "along-step", "move to interaction or boundary")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void AlongStepAction::step(CoreParams const& params, CoreStateHost& state) const
{
    auto propagate_and_update = [](CoreTrackView& track) {
        detail::PropagateExecutor{}(track);
        detail::AlongStepExecutor{}(track);
    };
    auto execute = make_active_volumetric_thread_executor(
        params.ptr<MemSpace::native>(), state.ptr(), propagate_and_update);
    return launch_action(state, execute);
}

#if !CELER_USE_DEVICE
void AlongStepAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
